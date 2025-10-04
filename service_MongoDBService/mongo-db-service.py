import asyncio
from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json
from datetime import datetime
import uuid
import hashlib
import os

import asyncio
import aio_pika


import sys
import os

# Import necessary MongoDB modules
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import DuplicateKeyError



class HTTP_SERVER():
    def __init__(self, httpServerHost, httpServerPort, httpServerPrivilegedIpAddress=["127.0.0.1"], data_class_instance=None):
        self.app = FastAPI()
        self.host = httpServerHost
        self.port = httpServerPort

        self.privilegedIpAddress = httpServerPrivilegedIpAddress

        #<HTTP_SERVER_CORS_ADDITION_START>
        self.app.add_middleware(CORSMiddleware, allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"],)
        #<HTTP_SERVER_CORS_ADDITION_END>

        self.data_class = data_class_instance  # Reference to the Data class instance

        # MongoDB connection setup
        self.mongo_client = MongoClient('mongodb://localhost:27017/', server_api=ServerApi('1'))
        self.db = self.mongo_client["neuralperk"]  # Database name from MongoSchema.json
        
        # Collections based on schema
        self.customers_collection = self.db["customers"]
        self.blender_objects_collection = self.db["blenderObjects"]
        self.sessions_collection = self.db["sessions"]
        
        # Apply schema validation
        self.apply_schema_validation()

    def apply_schema_validation(self):
        """Apply MongoDB schema validation to collections"""
        try:
            # Load schema from file
            schema_file_path = os.path.join(os.path.dirname(__file__), 'mongo-db-schema.json')
            with open(schema_file_path, 'r') as f:
                schema = json.load(f)
            
            # Apply validation to each collection
            for collection_config in schema['collections']:
                collection_name = collection_config['name']
                collection = self.db[collection_name]
                
                # Check if collection exists, if not create it
                if collection_name not in self.db.list_collection_names():
                    self.db.create_collection(collection_name)
                
                # Apply schema validation
                self.db.command({
                    'collMod': collection_name,
                    'validator': collection_config['validator'],
                    'validationLevel': collection_config['validationLevel'],
                    'validationAction': collection_config['validationAction']
                })
                
                print(f"Applied schema validation to collection: {collection_name}")
                
        except Exception as e:
            print(f"Warning: Could not apply schema validation: {str(e)}")
            print("Collections will use default validation")

    def parse_date_time(self,field):
        if isinstance(field, dict) and "$date" in field:
            return datetime.fromisoformat(field["$date"].replace("Z", "+00:00"))
        return field

    def generate_uuid(self):
        """Generate a unique UUID string"""
        return str(uuid.uuid4())

    def calculate_file_hash(self, file_path):
        """Calculate SHA-256 hash of the string value of file_path
        Args:
            file_path (str): Path to the file (as a string) to hash
        Returns:
            str: SHA-256 hash of the string value of file_path
        """
        try:
            if not isinstance(file_path, str):
                raise ValueError("file_path must be a string")
            hash_sha256 = hashlib.sha256()
            hash_sha256.update(file_path.encode("utf-8"))
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Error calculating hash for file path string '{file_path}': {str(e)}")
            return None

       
    async def configure_routes(self):
        # ========================================
        # ROOT ENDPOINT
        # ========================================
        @self.app.get("/api/mongodb-service/")
        async def root():
            """Root endpoint to check if MongoDB Service is active"""
            return JSONResponse(content={"message": "MongoDB Service is Active"}, status_code=200)
        
        # ========================================
        # CUSTOMER MANAGEMENT ENDPOINTS
        # ========================================
        
        @self.app.post("/api/mongodb-service/customers/add")
        async def add_customer(request: Request):
            """Add a new customer to the database
            Required fields: email, password
            Returns: Success message with customer details including auto-generated customerId
            """
            try:
                body = await request.json()
                
                # Validate required fields
                required_fields = ["email", "password"]
                for field in required_fields:
                    if field not in body:
                        raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
                
                # Check if customer already exists by email
                existing_customer = self.customers_collection.find_one({"email": body["email"]})
                if existing_customer:
                    raise HTTPException(status_code=409, detail="Customer with this email already exists")
                
                # Generate unique customer ID
                customer_id = self.generate_uuid()
                
                # Create customer document
                customer_doc = {
                    "customerId": customer_id,
                    "email": body["email"],
                    "password": body["password"]
                }
                
                result = self.customers_collection.insert_one(customer_doc)
                
                return JSONResponse(
                    content={
                        "message": "Customer added successfully",
                        "customerId": customer_id,
                        "email": body["email"]
                    },
                    status_code=201
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.get("/api/mongodb-service/customers/check/{customer_id}")
        async def check_customer_exists(customer_id: str):
            """Check if a customer exists in the database
            Parameters: customer_id (path parameter)
            Returns: Customer exists status and basic info if found
            """
            try:
                customer = self.customers_collection.find_one({"customerId": customer_id})
                
                if customer:
                    return JSONResponse(
                        content={
                            "exists": True,
                            "customerId": customer["customerId"],
                            "email": customer["email"]
                        },
                        status_code=200
                    )
                else:
                    return JSONResponse(
                        content={
                            "exists": False,
                            "message": "Customer not found"
                        },
                        status_code=404
                    )
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.get("/api/mongodb-service/customers/find-by-email/{email}")
        async def find_customer_by_email(email: str):
            """Find a customer by email address
            Parameters: email (path parameter)
            Returns: Customer details if found
            """
            try:
                customer = self.customers_collection.find_one({"email": email})
                
                if customer:
                    return JSONResponse(
                        content={
                            "exists": True,
                            "customerId": customer["customerId"],
                            "email": customer["email"],
                            "password": customer["password"]  # Include hashed password for auth verification
                        },
                        status_code=200
                    )
                else:
                    return JSONResponse(
                        content={
                            "exists": False,
                            "message": "Customer not found"
                        },
                        status_code=404
                    )
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        # ========================================
        # BLENDER OBJECT MANAGEMENT ENDPOINTS
        # ========================================
        
        @self.app.post("/api/mongodb-service/blender-objects/add")
        async def add_blender_object(request: Request):
            """Add a new blender object to the database
            Required fields: customerId
            Optional fields: blendFileName, blendFilePath, renderedVideoPath, isPaid, objectState, cost
            Note: cost field is only used when isPaid is true
            Returns: Success message with object details including auto-generated objectId
            """
            try:
                body = await request.json()
                
                # Validate required fields
                required_fields = ["customerId"]
                for field in required_fields:
                    if field not in body:
                        raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
                
                # Check if customer exists
                customer = self.customers_collection.find_one({"customerId": body["customerId"]})
                if not customer:
                    raise HTTPException(status_code=404, detail="Customer not found")
                
                # Generate unique object ID
                object_id = self.generate_uuid()

                # print(body["blendFilePath"])
                # print(body.get("blendFilePath"))
                
                # Calculate blend file hash if blendFilePath is provided
                blend_file_hash = None
                if body.get("blendFilePath"):
                    blend_file_hash = self.calculate_file_hash(body["blendFilePath"])
                
                # Create blender object document
                object_doc = {
                    "objectId": object_id,
                    "blendFileName": body.get("blendFileName", None),
                    "blendFilePath": body.get("blendFilePath", None),
                    "blendFileHash": blend_file_hash,
                    "renderedVideoPath": body.get("renderedVideoPath", None),
                    "customerId": body["customerId"],
                    "isPaid": body.get("isPaid", False),
                    "objectState": body.get("objectState", "ready-to-render"),
                    "cost": body.get("cost", None) if body.get("isPaid", False) else None
                }
                
                result = self.blender_objects_collection.insert_one(object_doc)
                
                return JSONResponse(
                    content={
                        "message": "Blender object added successfully",
                        "objectId": object_id,
                        "customerId": body["customerId"],
                        "blendFileName": object_doc["blendFileName"],
                    },
                    status_code=201
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.put("/api/mongodb-service/blender-objects/update-blend-file")
        async def update_blend_file_path(request: Request):
            """Update the blend file path for a blender object
            Required fields: objectId, customerId, blendFilePath
            Returns: Success message with updated object details including calculated blendFileHash
            """
            try:
                body = await request.json()
                
                # Validate required fields
                required_fields = ["objectId", "customerId", "blendFilePath"]
                for field in required_fields:
                    if field not in body:
                        raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
                
                # Check if customer exists
                customer = self.customers_collection.find_one({"customerId": body["customerId"]})
                if not customer:
                    raise HTTPException(status_code=404, detail="Customer not found")
                
                # Calculate SHA-256 hash of the new blend file
                blend_file_hash = self.calculate_file_hash(body["blendFilePath"])
                
                # Update the blend file path and hash
                result = self.blender_objects_collection.update_one(
                    {"objectId": body["objectId"], "customerId": body["customerId"]},
                    {
                        "$set": {
                            "blendFilePath": body["blendFilePath"],
                            "blendFileHash": blend_file_hash
                        }
                    }
                )
                
                if result.matched_count == 0:
                    raise HTTPException(status_code=404, detail="Blender object not found")
                
                return JSONResponse(
                    content={
                        "message": "Blend file path and hash updated successfully",
                        "objectId": body["objectId"],
                        "blendFilePath": body["blendFilePath"],
                    },
                    status_code=200
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.put("/api/mongodb-service/blender-objects/update-video-file")
        async def update_video_file_path(request: Request):
            """Update the rendered video path for a blender object
            Required fields: objectId, customerId, renderedVideoPath
            Returns: Success message with updated object details
            """
            try:
                body = await request.json()
                
                # Validate required fields
                required_fields = ["objectId", "customerId", "renderedVideoPath"]
                for field in required_fields:
                    if field not in body:
                        raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
                
                # Check if customer exists
                customer = self.customers_collection.find_one({"customerId": body["customerId"]})
                if not customer:
                    raise HTTPException(status_code=404, detail="Customer not found")
                
                # Update the video file path
                result = self.blender_objects_collection.update_one(
                    {"objectId": body["objectId"], "customerId": body["customerId"]},
                    {"$set": {"renderedVideoPath": body["renderedVideoPath"]}}
                )
                
                if result.matched_count == 0:
                    raise HTTPException(status_code=404, detail="Blender object not found")
                
                return JSONResponse(
                    content={
                        "message": "Video file path updated successfully",
                        "objectId": body["objectId"],
                        "renderedVideoPath": body["renderedVideoPath"]
                    },
                    status_code=200
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.put("/api/mongodb-service/blender-objects/update-cost")
        async def update_blender_object_cost(request: Request):
            """Update the cost for a blender object
            Required fields: objectId, customerId, cost
            Returns: Success message with updated object details
            """
            try:
                body = await request.json()
                
                # Validate required fields
                required_fields = ["objectId", "customerId", "cost"]
                for field in required_fields:
                    if field not in body:
                        raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
                
                # Validate cost is a number
                try:
                    cost_value = float(body["cost"])
                    if cost_value < 0:
                        raise HTTPException(status_code=400, detail="Cost must be a non-negative number")
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail="Cost must be a valid number")
                
                # Check if customer exists
                customer = self.customers_collection.find_one({"customerId": body["customerId"]})
                if not customer:
                    raise HTTPException(status_code=404, detail="Customer not found")
                
                # Check if blender object exists
                blender_object = self.blender_objects_collection.find_one({
                    "objectId": body["objectId"],
                    "customerId": body["customerId"]
                })
                if not blender_object:
                    raise HTTPException(status_code=404, detail="Blender object not found")
                
                # Update the cost
                result = self.blender_objects_collection.update_one(
                    {"objectId": body["objectId"], "customerId": body["customerId"]},
                    {"$set": {"cost": cost_value}}
                )
                
                if result.matched_count == 0:
                    raise HTTPException(status_code=404, detail="Blender object not found")
                
                return JSONResponse(
                    content={
                        "message": "Cost updated successfully",
                        "objectId": body["objectId"],
                        "customerId": body["customerId"],
                        "cost": cost_value
                    },
                    status_code=200
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        # ========================================
        # SESSION MANAGEMENT ENDPOINTS
        # ========================================
        
        @self.app.post("/api/mongodb-service/sessions/add")
        async def add_session(request: Request):
            """Add a new session to the database
            Required fields: customerId, blenderObjectId
            Optional fields: status (defaults to 'queued')
            Returns: Success message with session details including auto-generated sessionId
            """
            try:
                body = await request.json()
                
                # Validate required fields
                required_fields = ["customerId", "blenderObjectId"]
                for field in required_fields:
                    if field not in body:
                        raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
                
                # Check if customer exists
                customer = self.customers_collection.find_one({"customerId": body["customerId"]})
                if not customer:
                    raise HTTPException(status_code=404, detail="Customer not found")
                
                # Check if blender object exists
                blender_object = self.blender_objects_collection.find_one({"objectId": body["blenderObjectId"]})
                if not blender_object:
                    raise HTTPException(status_code=404, detail="Blender object not found")
                
                # Generate unique session ID
                session_id = self.generate_uuid()
                
                # Create session document with default status 'queued'
                session_doc = {
                    "sessionId": session_id,
                    "customerId": body["customerId"],
                    "blenderObjectId": body["blenderObjectId"],
                    "status": body.get("status", "queued")
                }
                
                result = self.sessions_collection.insert_one(session_doc)
                
                return JSONResponse(
                    content={
                        "message": "Session added successfully",
                        "sessionId": session_id,
                        "status": session_doc["status"]
                    },
                    status_code=201
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.put("/api/mongodb-service/sessions/update-status")
        async def update_session_status(request: Request):
            """Update the status of a session
            Required fields: sessionId, status
            Returns: Success message with updated session details
            """
            try:
                body = await request.json()
                
                # Validate required fields
                required_fields = ["sessionId", "status"]
                for field in required_fields:
                    if field not in body:
                        raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
                
                # Update the session status
                result = self.sessions_collection.update_one(
                    {"sessionId": body["sessionId"]},
                    {"$set": {"status": body["status"]}}
                )
                
                if result.matched_count == 0:
                    raise HTTPException(status_code=404, detail="Session not found")
                
                return JSONResponse(
                    content={
                        "message": "Session status updated successfully",
                        "sessionId": body["sessionId"],
                        "status": body["status"]
                    },
                    status_code=200
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        # ========================================
        # RENDERED IMAGES MANAGEMENT ENDPOINTS
        # ========================================
        
        @self.app.post("/api/mongodb-service/blender-objects/add-rendered-image")
        async def add_rendered_image(request: Request):
            """Add a single rendered image to a blender object
            Required fields: objectId, customerId, frameNumber, imageFilePath
            Returns: Success message with added image details
            """
            try:
                body = await request.json()
                
                # Validate required fields
                required_fields = ["objectId", "customerId", "frameNumber", "imageFilePath"]
                for field in required_fields:
                    if field not in body:
                        raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
                
                # Check if customer exists
                customer = self.customers_collection.find_one({"customerId": body["customerId"]})
                if not customer:
                    raise HTTPException(status_code=404, detail="Customer not found")
                
                # Check if blender object exists
                blender_object = self.blender_objects_collection.find_one({"objectId": body["objectId"]})
                if not blender_object:
                    raise HTTPException(status_code=404, detail="Blender object not found")
                
                # Check if frame number already exists for this object
                existing_frame = self.blender_objects_collection.find_one({
                    "objectId": body["objectId"],
                    "renderedImages.frameNumber": body["frameNumber"]
                })
                
                if existing_frame:
                    # Update existing frame
                    result = self.blender_objects_collection.update_one(
                        {
                            "objectId": body["objectId"],
                            "renderedImages.frameNumber": body["frameNumber"]
                        },
                        {
                            "$set": {
                                "renderedImages.$.imageFilePath": body["imageFilePath"]
                            }
                        }
                    )
                    action = "updated"
                else:
                    # Add new frame
                    result = self.blender_objects_collection.update_one(
                        {"objectId": body["objectId"], "customerId": body["customerId"]},
                        {
                            "$push": {
                                "renderedImages": {
                                    "frameNumber": body["frameNumber"],
                                    "imageFilePath": body["imageFilePath"]
                                }
                            }
                        }
                    )
                    action = "added"
                
                if result.matched_count == 0:
                    raise HTTPException(status_code=404, detail="Blender object not found")
                
                return JSONResponse(
                    content={
                        "message": f"Rendered image {action} successfully",
                        "objectId": body["objectId"],
                        "frameNumber": body["frameNumber"],
                        "imageFilePath": body["imageFilePath"],
                        "action": action
                    },
                    status_code=200
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.post("/api/mongodb-service/blender-objects/add-rendered-images-batch")
        async def add_rendered_images_batch(request: Request):
            """Add multiple rendered images to blender objects in batch
            Required fields: images (array of objects with objectId, customerId, frameNumber, imageFilePath)
            Returns: Success message with batch operation results
            """
            try:
                body = await request.json()
                
                # Validate required fields
                if "images" not in body or not isinstance(body["images"], list):
                    raise HTTPException(status_code=400, detail="Missing or invalid 'images' array field")
                
                if len(body["images"]) == 0:
                    raise HTTPException(status_code=400, detail="Images array cannot be empty")
                
                # Validate each image object
                for i, image in enumerate(body["images"]):
                    required_fields = ["objectId", "customerId", "frameNumber", "imageFilePath"]
                    for field in required_fields:
                        if field not in image:
                            raise HTTPException(status_code=400, detail=f"Image {i}: Missing required field: {field}")
                
                # Group images by objectId for efficient batch processing
                images_by_object = {}
                for image in body["images"]:
                    object_id = image["objectId"]
                    if object_id not in images_by_object:
                        images_by_object[object_id] = []
                    images_by_object[object_id].append(image)
                
                # Process each object
                results = []
                for object_id, images in images_by_object.items():
                    # Check if object exists
                    blender_object = self.blender_objects_collection.find_one({"objectId": object_id})
                    if not blender_object:
                        results.append({
                            "objectId": object_id,
                            "status": "failed",
                            "error": "Blender object not found"
                        })
                        continue
                    
                    # Check if customer exists
                    customer = self.customers_collection.find_one({"customerId": images[0]["customerId"]})
                    if not customer:
                        results.append({
                            "objectId": object_id,
                            "status": "failed",
                            "error": "Customer not found"
                        })
                        continue
                    
                    # Process images for this object
                    for image in images:
                        try:
                            # Check if frame already exists
                            existing_frame = self.blender_objects_collection.find_one({
                                "objectId": object_id,
                                "renderedImages.frameNumber": image["frameNumber"]
                            })
                            
                            if existing_frame:
                                # Update existing frame
                                self.blender_objects_collection.update_one(
                                    {
                                        "objectId": object_id,
                                        "renderedImages.frameNumber": image["frameNumber"]
                                    },
                                    {
                                        "$set": {
                                            "renderedImages.$.imageFilePath": image["imageFilePath"]
                                        }
                                    }
                                )
                                results.append({
                                    "objectId": object_id,
                                    "frameNumber": image["frameNumber"],
                                    "status": "updated"
                                })
                            else:
                                # Add new frame
                                self.blender_objects_collection.update_one(
                                    {"objectId": object_id, "customerId": image["customerId"]},
                                    {
                                        "$push": {
                                            "renderedImages": {
                                                "frameNumber": image["frameNumber"],
                                                "imageFilePath": image["imageFilePath"]
                                            }
                                        }
                                    }
                                )
                                results.append({
                                    "objectId": object_id,
                                    "frameNumber": image["frameNumber"],
                                    "status": "added"
                                })
                        except Exception as e:
                            results.append({
                                "objectId": object_id,
                                "frameNumber": image["frameNumber"],
                                "status": "failed",
                                "error": str(e)
                            })
                
                # Count successes and failures
                successful = len([r for r in results if r["status"] in ["added", "updated"]])
                failed = len([r for r in results if r["status"] == "failed"])
                
                return JSONResponse(
                    content={
                        "message": f"Batch operation completed: {successful} successful, {failed} failed",
                        "totalProcessed": len(body["images"]),
                        "successful": successful,
                        "failed": failed,
                        "results": results
                    },
                    status_code=200
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        # ========================================
        # BLEND FILE NAME MANAGEMENT ENDPOINTS
        # ========================================
        
        @self.app.put("/api/mongodb-service/blender-objects/update-blend-file-name")
        async def update_blend_file_name(request: Request):
            """Update the blend file name for a blender object
            Required fields: objectId, customerId, blendFileName
            Returns: Success message with updated object details
            """
            try:
                body = await request.json()
                
                # Validate required fields
                required_fields = ["objectId", "customerId", "blendFileName"]
                for field in required_fields:
                    if field not in body:
                        raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
                
                # Check if customer exists
                customer = self.customers_collection.find_one({"customerId": body["customerId"]})
                if not customer:
                    raise HTTPException(status_code=404, detail="Customer not found")
                
                # Update the blend file name
                result = self.blender_objects_collection.update_one(
                    {"objectId": body["objectId"], "customerId": body["customerId"]},
                    {"$set": {"blendFileName": body["blendFileName"]}}
                )
                
                if result.matched_count == 0:
                    raise HTTPException(status_code=404, detail="Blender object not found")
                
                return JSONResponse(
                    content={
                        "message": "Blend file name updated successfully",
                        "objectId": body["objectId"],
                        "blendFileName": body["blendFileName"]
                    },
                    status_code=200
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.get("/api/mongodb-service/blender-objects/get-blend-file-name/{object_id}")
        async def get_blend_file_name(object_id: str, customer_id: str):
            """Get the blend file name for a blender object
            Parameters: object_id (path parameter), customer_id (query parameter)
            Returns: Blend file name, object details, and cost
            """
            try:
                # Check if blender object exists
                blender_object = self.blender_objects_collection.find_one({
                    "objectId": object_id,
                    "customerId": customer_id
                })
                
                if not blender_object:
                    raise HTTPException(status_code=404, detail="Blender object not found")
                
                return JSONResponse(
                    content={
                        "objectId": object_id,
                        "customerId": customer_id,
                        "blendFileName": blender_object.get("blendFileName"),
                        "blendFilePath": blender_object.get("blendFilePath"),
                        "blendFileHash": blender_object.get("blendFileHash"),
                        "cost": blender_object.get("cost"),
                        "message": "Blend file name retrieved successfully"
                    },
                    status_code=200
                )
                    
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.get("/api/mongodb-service/blender-objects/find-by-hash/{file_hash}")
        async def find_blend_file_by_hash(file_hash: str):
            """Find blend file path by SHA-256 hash
            Parameters: file_hash (path parameter) - SHA-256 hash of the blend file
            Returns: Blend file path, object details, and cost
            """
            try:
                # Validate hash format (SHA-256 should be 64 characters)
                if len(file_hash) != 64 or not all(c in '0123456789abcdef' for c in file_hash.lower()):
                    raise HTTPException(status_code=400, detail="Invalid SHA-256 hash format")
                
                # Find blender object with matching hash
                blender_object = self.blender_objects_collection.find_one({
                    "blendFileHash": file_hash
                })
                
                if not blender_object:
                    raise HTTPException(status_code=404, detail="No blend file found with this hash")
                
                return JSONResponse(
                    content={
                        "objectId": blender_object["objectId"],
                        "customerId": blender_object["customerId"],
                        "blendFileName": blender_object.get("blendFileName"),
                        "blendFilePath": blender_object.get("blendFilePath"),
                        "cost": blender_object.get("cost"),
                        "message": "Blend file found successfully by hash"
                    },
                    status_code=200
                )
                    
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.get("/api/mongodb-service/blender-objects/check-plan/{object_id}")
        async def check_plan(object_id: str):
            """Check the payment plan status for a blender object
            Parameters: object_id (path parameter)
            Returns: Payment status (paid/unpaid), isPaid field, and cost
            """
            try:
                # Find blender object by objectId
                blender_object = self.blender_objects_collection.find_one({"objectId": object_id})
                
                if not blender_object:
                    raise HTTPException(status_code=404, detail="Blender object not found")
                
                # Check isPaid field and return appropriate status
                is_paid = blender_object.get("isPaid", False)
                payment_status = "paid" if is_paid else "unpaid"
                
                return JSONResponse(
                    content={
                        "objectId": object_id,
                        "paymentStatus": payment_status,
                        "isPaid": is_paid,
                        "cost": blender_object.get("cost"),
                        "message": f"Payment status: {payment_status}"
                    },
                    status_code=200
                )
                    
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
     
        @self.app.get("/api/mongodb-service/blender-objects/get-blend-file-from-path")
        async def get_blend_file_from_path(file_path: str):
            """Get blend file path from database by calculating hash of input path and finding matching record
            Parameters: file_path (query parameter) - File path to search for in database
            Returns: Blend file details and cost if found in database
            """
            try:
                if not file_path:
                    raise HTTPException(status_code=400, detail="file_path query parameter is required")
                
                # Calculate hash of the input file path
                input_hash = self.calculate_file_hash(file_path)
                
                # Search in database for any record with matching hash
                blender_object = self.blender_objects_collection.find_one({
                    "blendFileHash": input_hash
                })
                
                if not blender_object:
                    raise HTTPException(status_code=404, detail="No blend file found with this path hash")
                
                return JSONResponse(
                    content={
                        "objectId": blender_object["objectId"],
                        "customerId": blender_object["customerId"],
                        "blendFileName": blender_object.get("blendFileName"),
                        "blendFilePath": blender_object.get("blendFilePath"),
                        "cost": blender_object.get("cost"),
                        "message": "Blend file found successfully by path hash"
                    },
                    status_code=200
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.get("/api/mongodb-service/blender-objects/get-rendered-images/{object_id}")
        async def get_rendered_images(object_id: str, customer_id: str):
            """Get all rendered images for a blender object
            Parameters: object_id (path parameter), customer_id (query parameter)
            Returns: List of rendered images with frame numbers and file paths
            """
            try:
                # Check if blender object exists and belongs to customer
                blender_object = self.blender_objects_collection.find_one({
                    "objectId": object_id,
                    "customerId": customer_id
                })
                
                if not blender_object:
                    raise HTTPException(status_code=404, detail="Blender object not found")
                
                # Get rendered images array
                rendered_images = blender_object.get("renderedImages", [])
                
                # Sort by frame number for consistent ordering
                rendered_images.sort(key=lambda x: x.get("frameNumber", 0))
                
                return JSONResponse(
                    content={
                        "objectId": object_id,
                        "customerId": customer_id,
                        "renderedImages": rendered_images,
                        "totalFrames": len(rendered_images),
                        "message": "Rendered images retrieved successfully"
                    },
                    status_code=200
                )
                    
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.get("/api/mongodb-service/blender-objects/get-by-object-id/{object_id}")
        async def get_blender_object_by_object_id(object_id: str, customer_id: str):
            """Get a blender object by object ID
            Parameters: object_id (path parameter), customer_id (query parameter)
            Returns: Complete blender object details
            """
            try:
                # Check if blender object exists and belongs to customer
                blender_object = self.blender_objects_collection.find_one({
                    "objectId": object_id,
                    "customerId": customer_id
                })
                
                if not blender_object:
                    raise HTTPException(status_code=404, detail="Blender object not found")
                
                # Convert ObjectId to string for JSON serialization
                if "_id" in blender_object:
                    blender_object["_id"] = str(blender_object["_id"])
                
                return JSONResponse(
                    content={
                        "objectId": object_id,
                        "customerId": customer_id,
                        "blenderObject": blender_object,
                        "message": "Blender object retrieved successfully"
                    },
                    status_code=200
                )
                    
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.get("/api/mongodb-service/blender-objects/get-by-customer/{customer_id}")
        async def get_blender_objects_by_customer(customer_id: str):
            """Get all blender objects associated with a specific customer
            Parameters: customer_id (path parameter)
            Returns: List of blender objects with objectId, blendFileName, isPaid, objectState, and cost
            """
            try:
                # Check if customer exists
                customer = self.customers_collection.find_one({"customerId": customer_id})
                if not customer:
                    raise HTTPException(status_code=404, detail="Customer not found")
                
                # Find all blender objects for this customer
                blender_objects = list(self.blender_objects_collection.find(
                    {"customerId": customer_id},
                    {"objectId": 1, "blendFileName": 1, "isPaid": 1, "objectState": 1, "cost": 1, "_id": 0}
                ))
                
                # Format the response
                objects_list = []
                for obj in blender_objects:
                    objects_list.append({
                        "objectId": obj["objectId"],
                        "blendFileName": obj.get("blendFileName"),
                        "isPaid": obj.get("isPaid", False),
                        "objectState": obj.get("objectState", "ready-to-render"),
                        "cost": obj.get("cost")
                    })
                
                return JSONResponse(
                    content={
                        "customerId": customer_id,
                        "blenderObjects": objects_list,
                        "totalObjects": len(objects_list),
                        "message": "Blender objects retrieved successfully"
                    },
                    status_code=200
                )
                    
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.put("/api/mongodb-service/blender-objects/change-state")
        async def change_object_state(request: Request):
            """Change the state of a blender object
            Required fields: objectId, customerId, objectState
            Returns: Success message with updated object details
            """
            try:
                body = await request.json()
                
                # Validate required fields
                required_fields = ["objectId", "customerId", "objectState"]
                for field in required_fields:
                    if field not in body:
                        raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
                
                # Validate objectState value
                valid_states = ["ready-to-render", "processing", "video-ready"]
                if body["objectState"] not in valid_states:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid objectState. Must be one of: {', '.join(valid_states)}"
                    )
                
                # Check if customer exists
                customer = self.customers_collection.find_one({"customerId": body["customerId"]})
                if not customer:
                    raise HTTPException(status_code=404, detail="Customer not found")
                
                # Update the object state
                result = self.blender_objects_collection.update_one(
                    {"objectId": body["objectId"], "customerId": body["customerId"]},
                    {"$set": {"objectState": body["objectState"]}}
                )
                
                if result.matched_count == 0:
                    raise HTTPException(status_code=404, detail="Blender object not found")
                
                return JSONResponse(
                    content={
                        "message": "Object state updated successfully",
                        "objectId": body["objectId"],
                        "customerId": body["customerId"],
                        "objectState": body["objectState"],
                        "success": True
                    },
                    status_code=200
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.delete("/api/mongodb-service/blender-objects/delete/{object_id}")
        async def delete_blender_object(object_id: str, customer_id: str):
            """Delete a blender object by objectId and customerId
            Parameters: object_id (path parameter), customer_id (query parameter)
            Returns: Success message with deleted object details
            """
            try:
                # Check if customer exists
                customer = self.customers_collection.find_one({"customerId": customer_id})
                if not customer:
                    raise HTTPException(status_code=404, detail="Customer not found")
                
                # Check if blender object exists and belongs to customer
                blender_object = self.blender_objects_collection.find_one({
                    "objectId": object_id,
                    "customerId": customer_id
                })
                
                if not blender_object:
                    raise HTTPException(status_code=404, detail="Blender object not found")
                
                # Delete the blender object
                result = self.blender_objects_collection.delete_one({
                    "objectId": object_id,
                    "customerId": customer_id
                })
                
                if result.deleted_count == 0:
                    raise HTTPException(status_code=404, detail="Blender object not found or already deleted")
                
                return JSONResponse(
                    content={
                        "message": "Blender object deleted successfully",
                        "objectId": object_id,
                        "customerId": customer_id,
                        "blendFileName": blender_object.get("blendFileName"),
                        "deletedCount": result.deleted_count
                    },
                    status_code=200
                )
                    
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def run_app(self):
        config = uvicorn.Config(self.app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()

class Data():
    def __init__(self):
        self.value = None

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value


class Service():
    def __init__(self, httpServer = None):
        self.httpServer = httpServer

    async def startService(self):
        await self.httpServer.configure_routes()
        await self.httpServer.run_app()

async def start_service():
    dataClass = Data()

    httpServerPort = 12000
    httpServerHost = "127.0.0.1"
    httpServerPrivilegedIpAddress = ["127.0.0.1"]
    
    http_server = HTTP_SERVER(httpServerHost=httpServerHost, httpServerPort=httpServerPort, httpServerPrivilegedIpAddress=httpServerPrivilegedIpAddress, data_class_instance=dataClass)


    service = Service(http_server)
    await service.startService()

if __name__ == "__main__":
    asyncio.run(start_service())