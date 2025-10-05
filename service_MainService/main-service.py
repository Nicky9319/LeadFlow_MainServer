import asyncio
import logging
import traceback

from fastapi import FastAPI, Response, Request, HTTPException, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import BackgroundTasks

from fastapi.responses import JSONResponse

import uvicorn
import httpx
import json
from datetime import datetime
import uuid
import hashlib
import secrets
import base64
from typing import Optional, Dict, Any, List

import asyncio

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_core.messages import BaseMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

import sys
import os
import io

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security scheme for token validation
security = HTTPBearer(auto_error=False)

class LeadInput(BaseModel):
    """Input schema for adding a lead to CRM"""
    url: str = Field(description="The URL or social media profile of the lead")
    username: str = Field(default="", description="The username or handle of the person")
    platform: str = Field(default="", description="The platform where this lead was found (LinkedIn, Twitter, Instagram, etc.)")
    status: str = Field(default="Cold Message", description="Lead status")
    bucket_id: str = Field(default="", description="The bucket to store the lead in")
    notes: str = Field(default="", description="Any additional notes about the lead")

class LeadExtractionAgent():
    def __init__(self, mongodb_service_url: str):
        print(f"Initializing LeadExtractionAgent with MongoDB URL: {mongodb_service_url}")
        self.mongodb_service_url = mongodb_service_url
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
        # Initialize OpenAI LLM
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        print("Initializing OpenAI ChatGPT model...")
        try:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",  # Using mini model which is cheaper and faster
                api_key=openai_api_key,
                temperature=0.1,
                max_tokens=500,  # Reduced to prevent token limit issues
                request_timeout=60  # Add timeout for stability
            )
            print("OpenAI model initialized successfully with gpt-4o")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI model: {e}")
            logger.error(f"OpenAI initialization error traceback: {traceback.format_exc()}")
            raise
        
        # Create tools for the agent
        print("Creating agent tools...")
        try:
            self.tools = self._create_tools()
            print(f"Created {len(self.tools)} tools for the agent")
        except Exception as e:
            logger.error(f"Failed to create agent tools: {e}")
            raise
        
        # Create prompt template
        print("Creating prompt template...")
        try:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a lead extraction agent. Your job is to analyze images that might contain lead information (business cards, LinkedIn profiles, contact information, etc.).
                
                        You have access to a tool called 'add_lead_to_crm' that can save lead information to the CRM system. This tool accepts structured input with the following fields:
                        - url (required): The URL or social media profile of the lead
                        - username (optional): The username or handle of the person  
                        - platform (optional): The platform where this lead was found (LinkedIn, Twitter, Instagram, etc.)
                        - status (optional): Lead status, defaults to 'Cold Message'
                        - bucket_id (optional): The bucket to store the lead in (will be provided in the context)
                        - notes (optional): Any additional notes about the lead
                                        
                        When you receive an image:
                        1. Carefully analyze it to see if it contains any lead-related information such as:
                        - Names of people or businesses
                        - Contact information (email, phone, social media)
                        - Professional titles or roles
                        - Company names
                        - Social media profiles (LinkedIn, Twitter, Instagram, etc.)
                        - Any other business-related contact information

                        2. If you find lead information, extract the relevant details and use the add_lead_to_crm tool with the appropriate structured fields.
                        3. If the image doesn't contain any lead information, simply respond that no lead information was found.

                        Be thorough in your analysis but only extract information that is clearly visible and relevant for lead generation."""),
                
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            print("Prompt template created successfully")
        except Exception as e:
            logger.error(f"Failed to create prompt template: {e}")
            raise
        
        # Create the agent
        print("Creating OpenAI functions agent...")
        try:
            self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                return_intermediate_steps=True,
                max_iterations=3
            )
            print("Agent and executor created successfully")
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise
    
    def _create_tools(self) -> List[Tool]:
        """Create the tools available to the agent"""
        
        async def add_lead_to_crm_impl(
            url: str,
            username: str = "",
            platform: str = "",
            status: str = "Cold Message",
            bucket_id: str = "",
            notes: str = ""
        ) -> str:
            """Add a lead to the CRM system"""
            print(f"add_lead_to_crm_impl called with url={url}, username={username}, platform={platform}, status={status}, bucket_id={bucket_id}, notes={notes}")
            try:
                # Create lead data dict
                lead_data = {
                    "url": url,
                    "username": username,
                    "platform": platform,
                    "status": status,
                    "bucket_id": bucket_id,
                    "notes": notes
                }
                print(f"Lead data: {lead_data}")
                
                # Validate required fields
                if not url:
                    logger.warning("No URL provided in lead data")
                    return "Error: URL is required for lead creation"
                
                if not bucket_id:
                    # Default to a bucket if not specified - you might want to handle this differently
                    lead_data["bucket_id"] = "default-bucket"
                    print("No bucket_id provided, using default-bucket")
                
                # Call MongoDB service to add the lead using async httpx
                request_data = {
                    "url": lead_data.get("url", ""),
                    "username": lead_data.get("username", ""),
                    "platform": lead_data.get("platform", ""),
                    "status": lead_data.get("status", "Cold Message"),
                    "bucket_id": lead_data.get("bucket_id"),
                    "notes": lead_data.get("notes", "")
                }
                print(f"Making request to MongoDB service: {self.mongodb_service_url}/api/mongodb-service/leads/add-lead")
                print(f"Request data: {request_data}")
                
                response = await self.http_client.post(
                    f"{self.mongodb_service_url}/api/mongodb-service/leads/add-lead",
                    json=request_data
                )
                print(f"MongoDB service response status: {response.status_code}")
                print(f"MongoDB service response text: {response.text}")
                
                if response.status_code == 201:
                    result = response.json()
                    lead_info = result.get('lead', {})
                    # Set a flag to indicate successful lead creation
                    self._last_successful_lead = lead_info
                    return json.dumps({
                        "success": True,
                        "message": f"Successfully added lead to CRM. Lead ID: {lead_info.get('leadId', 'Unknown')}",
                        "lead_data": lead_info
                    })
                else:
                    return json.dumps({
                        "success": False,
                        "message": f"Error adding lead to CRM: {response.text}",
                        "error": response.text
                    })
                    
            except Exception as e:
                return f"Error adding lead to CRM: {str(e)}"
        
        return [
            StructuredTool.from_function(
                name="add_lead_to_crm",
                description="Add a lead to the CRM system with structured data including URL, username, platform, status, bucket_id, and notes",
                func=add_lead_to_crm_impl,
                coroutine=add_lead_to_crm_impl  # For async support
            )
        ]
    
    async def process_image_from_memory(self, image_content: bytes, bucket_id: str = None, filename: str = "uploaded_image") -> Dict[str, Any]:
        """Process an image from memory and extract lead information if present"""
        print(f"Processing image: {filename} (in memory) with bucket_id: {bucket_id}")
        try:
            print(f"Processing image from memory, size: {len(image_content)} bytes")
                
            # Check image size and resize if too large
            max_size = 20 * 1024 * 1024  # 20MB limit for OpenAI
            if len(image_content) > max_size:
                logger.warning(f"Image size {len(image_content)} bytes exceeds limit, need to resize")
                # For now, we'll try to resize using PIL
                from PIL import Image
                import io
                
                pil_image = Image.open(io.BytesIO(image_content))
                # Resize to maximum dimension of 2048 pixels
                max_dimension = 2048
                if max(pil_image.size) > max_dimension:
                    ratio = max_dimension / max(pil_image.size)
                    new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                    
                # Save resized image back to bytes
                output_buffer = io.BytesIO()
                pil_image.save(output_buffer, format='PNG', optimize=True)
                image_content = output_buffer.getvalue()
                print(f"Image resized to {len(image_content)} bytes")
                
            image_data = base64.b64encode(image_content).decode('utf-8')
            print(f"Image encoded successfully, size: {len(image_data)} characters")
            
            # Create the message with image
            print("Creating message with image for agent processing")
            messages = [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": f"Please analyze this image for any lead information. If you find any, extract it and add it to the CRM using bucket_id: {bucket_id or 'default-bucket'}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                )
            ]
            
            # Run the agent
            print("Starting agent execution...")
            try:
                result = await self.agent_executor.ainvoke({
                    "input": f"Analyze the provided image for lead information. If you find any lead information, use the add_lead_to_crm tool with bucket_id: '{bucket_id or 'default-bucket'}' to save the lead data.",
                    "chat_history": messages
                })
                print(f"Agent execution completed successfully")
                print(f"Agent result: {result.get('output', 'No output')[:200]}...")  # Log first 200 chars
                
                return {
                    "success": True,
                    "result": result["output"],
                    "intermediate_steps": result.get("intermediate_steps", [])
                }
            except Exception as agent_error:
                logger.error(f"Agent execution failed: {agent_error}")
                logger.error(f"Agent error traceback: {traceback.format_exc()}")
                
                # Check if any leads were successfully created despite the error
                # by checking if our tool was called successfully in the process
                successful_leads = []
                error_msg = str(agent_error)
                
                # If it's an OpenAI API error but we can see tool execution in logs
                if "openai" in error_msg.lower() and hasattr(self, '_last_successful_lead'):
                    print("OpenAI API error occurred, but checking if lead was created...")
                    return {
                        "success": True,  # Consider it successful if lead was created
                        "result": f"Lead information was successfully extracted and saved to CRM, but there was an issue with the AI response generation. Lead creation was successful.",
                        "intermediate_steps": [],
                        "warning": "OpenAI API error after successful lead creation"
                    }
                
                raise agent_error
            
        except Exception as e:
            logger.error(f"Error in process_image_from_memory: {e}")
            logger.error(f"Process image error traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "result": f"Error processing image: {str(e)}"
            }
    
    async def process_image_background(self, image_content: bytes, bucket_id: str, filename: str, request_id: str = None):
        """Process image in background and log results"""
        try:
            print(f"Starting background processing for {filename} (request_id: {request_id})")
            result = await self.process_image_from_memory(image_content, bucket_id, filename)
            
            if result["success"]:
                print(f"Background processing completed successfully for {filename} (request_id: {request_id})")
                print(f"Agent result: {result.get('result', 'No result')}")
                
                # Log any leads that were extracted
                extracted_leads_count = 0
                for step in result.get("intermediate_steps", []):
                    if len(step) >= 2:
                        action, observation = step[0], step[1]
                        if hasattr(action, 'tool') and action.tool == "add_lead_to_crm":
                            try:
                                tool_result = json.loads(observation)
                                if tool_result.get("success"):
                                    extracted_leads_count += 1
                            except (json.JSONDecodeError, AttributeError):
                                if "Successfully added lead to CRM" in str(observation):
                                    extracted_leads_count += 1
                
                print(f"Background processing extracted {extracted_leads_count} leads for {filename} (request_id: {request_id})")
            else:
                logger.error(f"Background processing failed for {filename} (request_id: {request_id}): {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error in background processing for {filename} (request_id: {request_id}): {e}")
            logger.error(f"Background processing error traceback: {traceback.format_exc()}")

    async def close(self):
        """Clean up resources"""
        await self.http_client.aclose()
    
    

class HTTP_SERVER():
    def __init__(self, httpServerHost, httpServerPort, httpServerPrivilegedIpAddress=["127.0.0.1"], data_class_instance=None):
        self.app = FastAPI()
        self.host = httpServerHost
        self.port = httpServerPort

        self.privilegedIpAddress = httpServerPrivilegedIpAddress        
        self.app.add_middleware(CORSMiddleware, allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"],)
        
        self.data_class = data_class_instance  # Reference to the Data class instance
        
        # Get MongoDB service URL from environment
        env_url = os.getenv("MONGODB_SERVICE", "").strip()
        if not env_url or not (env_url.startswith("http://") or env_url.startswith("https://")):
            self.mongodb_service_url = "http://127.0.0.1:10000"
        else:
            self.mongodb_service_url = env_url
        
        # HTTP client for making requests to MongoDB service and Auth service
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Initialize the Lead Extraction Agent
        print("Initializing Lead Extraction Agent...")
        try:
            self.lead_agent = LeadExtractionAgent(self.mongodb_service_url)
            print("Lead Extraction Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LeadExtractionAgent: {e}")
            logger.error(f"Agent initialization error traceback: {traceback.format_exc()}")
            self.lead_agent = None

    async def configure_routes(self):

        @self.app.get("/api/main-service/")
        async def ping_main_service(request: Request):
            print("Ping received at main-service")
            return JSONResponse(content={"message": "Main service is active"}, status_code=200)
        # -------------------------
        # Bucket endpoints (proxy to MongoDB service)
        # -------------------------
        @self.app.post("/api/main-service/buckets/add-bucket")
        async def add_new_bucket(request: Request, bucket_name: str | None = None):
            # accept via query param or JSON body
            if not bucket_name:
                try:
                    body = await request.json()
                    bucket_name = body.get("bucket_name") or body.get("bucketName")
                except Exception:
                    bucket_name = None

            if not bucket_name:
                raise HTTPException(status_code=400, detail="bucket_name is required")

            try:
                resp = await self.http_client.post(
                    f"{self.mongodb_service_url}/api/mongodb-service/buckets/add-bucket",
                    json={"bucket_name": bucket_name},
                )
            except httpx.RequestError as e:
                raise HTTPException(status_code=503, detail=f"MongoDB service unreachable: {e}")

            try:
                content = resp.json()
            except Exception:
                content = {"detail": resp.text}

            return JSONResponse(status_code=resp.status_code, content=content)

        @self.app.get("/api/main-service/buckets/get-all-buckets")
        async def get_all_buckets():
            try:
                resp = await self.http_client.get(
                    f"{self.mongodb_service_url}/api/mongodb-service/buckets/get-all-buckets",
                )
            except httpx.RequestError as e:
                raise HTTPException(status_code=503, detail=f"MongoDB service unreachable: {e}")

            try:
                content = resp.json()
            except Exception:
                content = {"detail": resp.text}

            return JSONResponse(status_code=resp.status_code, content=content)

        @self.app.delete("/api/main-service/buckets/delete-bucket")
        async def delete_bucket(bucket_id: str | None = None):
            if not bucket_id:
                raise HTTPException(status_code=400, detail="bucket_id is required")
            try:
                resp = await self.http_client.delete(
                    f"{self.mongodb_service_url}/api/mongodb-service/buckets/delete-bucket",
                    params={"bucket_id": bucket_id},
                )
            except httpx.RequestError as e:
                raise HTTPException(status_code=503, detail=f"MongoDB service unreachable: {e}")

            try:
                content = resp.json()
            except Exception:
                content = {"detail": resp.text}

            return JSONResponse(status_code=resp.status_code, content=content)

        @self.app.put("/api/main-service/buckets/update-bucket-name")
        async def update_bucket_name(request: Request, bucket_id: str | None = None, bucket_name: str | None = None):
            # accept via query params or JSON body
            if not bucket_id or not bucket_name:
                try:
                    body = await request.json()
                    bucket_id = bucket_id or body.get("bucket_id") or body.get("bucketId")
                    bucket_name = bucket_name or body.get("bucket_name") or body.get("bucketName")
                except Exception:
                    pass

            if not bucket_id or not bucket_name:
                raise HTTPException(status_code=400, detail="bucket_id and bucket_name are required")

            try:
                resp = await self.http_client.put(
                    f"{self.mongodb_service_url}/api/mongodb-service/buckets/update-bucket-name",
                    json={"bucket_id": bucket_id, "bucket_name": bucket_name},
                )
            except httpx.RequestError as e:
                raise HTTPException(status_code=503, detail=f"MongoDB service unreachable: {e}")

            try:
                content = resp.json()
            except Exception:
                content = {"detail": resp.text}

            return JSONResponse(status_code=resp.status_code, content=content)

        # -------------------------
        # Leads endpoints (proxy to MongoDB service)
        # -------------------------
        @self.app.get("/api/main-service/leads/get-all-leads")
        async def get_all_leads(bucket_id: str | None = None):
            params = {}
            if bucket_id:
                params["bucket_id"] = bucket_id
            try:
                resp = await self.http_client.get(
                    f"{self.mongodb_service_url}/api/mongodb-service/leads/get-all-leads",
                    params=params,
                )
            except httpx.RequestError as e:
                raise HTTPException(status_code=503, detail=f"MongoDB service unreachable: {e}")

            try:
                content = resp.json()
            except Exception:
                content = {"detail": resp.text}

            return JSONResponse(status_code=resp.status_code, content=content)

        @self.app.post("/api/main-service/leads/add-lead")
        async def add_lead(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...), bucket_id: str = None):
            """
            Accepts a single uploaded file (image) and processes it in the background using the LeadExtractionAgent.
            Returns immediate response while processing happens in background.
            """
            print(f"Received add_lead request with file: {file.filename if file else 'None'}, bucket_id: {bucket_id}")
            
            if not self.lead_agent:
                logger.error("Lead extraction agent is not available")
                raise HTTPException(status_code=503, detail="Lead extraction agent is not available")
            
            # Generate unique request ID for tracking
            request_id = str(uuid.uuid4())[:8]
            print(f"Generated request_id: {request_id} for file: {file.filename}")
            
            # Get bucket_id from query params or form data
            if not bucket_id:
                try:
                    form_data = await request.form()
                    bucket_id = form_data.get("bucket_id") or form_data.get("bucketId")
                    print(f"Retrieved bucket_id from form data: {bucket_id}")
                except Exception as e:
                    logger.warning(f"Could not get form data: {e}")
                    pass
            
            # If still no bucket_id, try to get default bucket or create one
            if not bucket_id:
                print("No bucket_id provided, attempting to get or create default bucket")
                try:
                    # Try to get existing buckets
                    print("Fetching existing buckets from MongoDB service")
                    resp = await self.http_client.get(
                        f"{self.mongodb_service_url}/api/mongodb-service/buckets/get-all-buckets"
                    )
                    print(f"Get buckets response status: {resp.status_code}")
                    
                    if resp.status_code == 200:
                        buckets = resp.json().get("buckets", [])
                        print(f"Found {len(buckets)} existing buckets")
                        if buckets:
                            bucket_id = buckets[0]["bucketId"]  # Use first available bucket
                            print(f"Using existing bucket: {bucket_id}")
                        else:
                            # Create a default bucket
                            print("No existing buckets found, creating default bucket")
                            create_resp = await self.http_client.post(
                                f"{self.mongodb_service_url}/api/mongodb-service/buckets/add-bucket",
                                json={"bucket_name": "Default Leads"}
                            )
                            print(f"Create bucket response status: {create_resp.status_code}")
                            if create_resp.status_code == 201:
                                bucket_id = create_resp.json().get("bucket", {}).get("bucketId")
                                print(f"Created new bucket: {bucket_id}")
                except Exception as e:
                    logger.error(f"Error handling bucket: {e}")
                    logger.error(f"Bucket handling error traceback: {traceback.format_exc()}")
                    bucket_id = "default-bucket"  # Fallback
                    print(f"Using fallback bucket_id: {bucket_id}")
            
            try:
                # Read the uploaded file into memory
                print(f"Reading uploaded file: {file.filename}")
                image_content = await file.read()
                print(f"File read successfully, size: {len(image_content)} bytes")
                
                # Validate file is not empty
                if len(image_content) == 0:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty")
                
                # Schedule background processing
                print(f"Scheduling background processing for {file.filename} (request_id: {request_id})")
                background_tasks.add_task(
                    self.lead_agent.process_image_background,
                    image_content,
                    bucket_id,
                    file.filename,
                    request_id
                )
                
                # Return immediate response
                response_content = {
                    "message": "Image uploaded successfully and is being processed in the background",
                    "request_id": request_id,
                    "bucket_id": bucket_id,
                    "filename": file.filename,
                    "file_size": len(image_content),
                    "status": "processing",
                    "note": "Lead extraction will happen in the background. Check logs for processing results."
                }
                
                print(f"Returning immediate response for request_id: {request_id}")
                return JSONResponse(status_code=202, content=response_content)
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Exception in add_lead endpoint: {e}")
                logger.error(f"Add lead error traceback: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {e}")

        @self.app.put("/api/main-service/leads/update-lead-status")
        async def update_lead_status(request: Request, lead_id: str | None = None, status: str | None = None):
            if not lead_id or not status:
                try:
                    body = await request.json()
                    lead_id = lead_id or body.get("lead_id") or body.get("leadId")
                    status = status or body.get("status")
                except Exception:
                    pass

            if not lead_id or not status:
                raise HTTPException(status_code=400, detail="lead_id and status are required")

            try:
                resp = await self.http_client.put(
                    f"{self.mongodb_service_url}/api/mongodb-service/leads/update-lead-status",
                    json={"lead_id": lead_id, "status": status},
                )
            except httpx.RequestError as e:
                print("Error In Making Request to MongoDB Service:", e)
                raise HTTPException(status_code=503, detail=f"MongoDB service unreachable: {e}")

            try:
                content = resp.json()
            except Exception:
                content = {"detail": resp.text}

            return JSONResponse(status_code=resp.status_code, content=content)

        @self.app.get("/api/main-service/leads/processing-status/{request_id}")
        async def get_processing_status(request_id: str):
            """
            Get the status of background processing for a given request_id.
            Note: This is a simple implementation. For production, you might want to use Redis or database to track status.
            """
            # For now, this is a placeholder endpoint that acknowledges the request_id
            # In a production environment, you'd track processing status in a database or cache
            return JSONResponse(
                status_code=200,
                content={
                    "request_id": request_id,
                    "message": "Processing status tracking is available through application logs. Check server logs for detailed processing results.",
                    "note": "For real-time status tracking, consider implementing a database-backed status system."
                }
            )

        @self.app.put("/api/main-service/leads/update-lead-notes")
        async def update_lead_notes(request: Request, lead_id: str | None = None, notes: str | None = None):
            if not lead_id or notes is None:
                try:
                    body = await request.json()
                    lead_id = lead_id or body.get("lead_id") or body.get("leadId")
                    notes = notes or body.get("notes")
                except Exception:
                    pass

            if not lead_id:
                raise HTTPException(status_code=400, detail="lead_id is required")

            try:
                resp = await self.http_client.put(
                    f"{self.mongodb_service_url}/api/mongodb-service/leads/update-lead-notes",
                    json={"lead_id": lead_id, "notes": notes},
                )
            except httpx.RequestError as e:
                raise HTTPException(status_code=503, detail=f"MongoDB service unreachable: {e}")

            try:
                content = resp.json()
            except Exception:
                content = {"detail": resp.text}

            return JSONResponse(status_code=resp.status_code, content=content)
    
    async def cleanup(self):
        """Clean up resources"""
        await self.http_client.aclose()
        if self.lead_agent:
            await self.lead_agent.close()

        
        
    async def run_app(self):
        try:
            config = uvicorn.Config(self.app, host=self.host, port=self.port)
            server = uvicorn.Server(config)
            await server.serve()
        finally:
            await self.cleanup()

class Data():
    def __init__(self):
        pass

class Service():
    def __init__(self, httpServer = None):
        self.httpServer = httpServer

    async def startService(self):
        await self.httpServer.configure_routes()
        await self.httpServer.run_app()

        
async def start_service():
    dataClass = Data()

    #<HTTP_SERVER_INSTANCE_INTIALIZATION_START>

    #<HTTP_SERVER_PORT_START>
    httpServerPort = 8000
    #<HTTP_SERVER_PORT_END>

    #<HTTP_SERVER_HOST_START>
    httpServerHost = "0.0.0.0"
    #<HTTP_SERVER_HOST_END>

    #<HTTP_SERVER_PRIVILEGED_IP_ADDRESS_START>
    httpServerPrivilegedIpAddress = ["127.0.0.1"]
    #<HTTP_SERVER_PRIVILEGED_IP_ADDRESS_END>

    http_server = HTTP_SERVER(httpServerHost=httpServerHost, httpServerPort=httpServerPort, httpServerPrivilegedIpAddress=httpServerPrivilegedIpAddress, data_class_instance=dataClass)
    #<HTTP_SERVER_INSTANCE_INTIALIZATION_END>

    service = Service(http_server)
    await service.startService()

if __name__ == "__main__":
    asyncio.run(start_service())