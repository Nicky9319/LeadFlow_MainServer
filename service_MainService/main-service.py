import asyncio

from fastapi import FastAPI, Response, Request, HTTPException, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from fastapi.responses import JSONResponse

import uvicorn
import httpx
import json
from datetime import datetime
import uuid
import hashlib
import secrets

import asyncio


import sys
import os

from dotenv import load_dotenv
load_dotenv()

# Security scheme for token validation
security = HTTPBearer(auto_error=False)

class Lead_Manager():
    def __init__(self):
        pass
    
    def extract_lead_info_from_image(self):
        pass
    
    

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
        async def add_lead(file: UploadFile = File(...)):
            """
            Accepts a single uploaded file (image) and saves it locally as `image.png`.
            This endpoint does not contact the MongoDB service.
            """
            try:
                contents = await file.read()
                save_dir = os.path.dirname(__file__)
                save_path = os.path.join(save_dir, "image.png")
                with open(save_path, "wb") as f:
                    f.write(contents)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

            return JSONResponse(status_code=201, content={"message": "File saved", "path": "service_MainService/image.png"})

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
    
        
        
    async def run_app(self):
        config = uvicorn.Config(self.app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()

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