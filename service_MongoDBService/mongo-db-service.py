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
        self.db = self.mongo_client["leadflow"]  # Database name from MongoSchema.json
        
        # Collections based on schema
        self.buckets_collection = self.db["buckets"]
        self.leads_collection = self.db["leads"]
        # Ensure unique indexes for bucketId and leadId
        try:
            self.buckets_collection.create_index("bucketId", unique=True)
            self.leads_collection.create_index("leadId", unique=True)
        except Exception:
            # ignore errors creating indexes if already exist or server settings
            pass
    async def configure_routes(self):
        # -------------------------
        # Health Check Endpoint
        # -------------------------
        @self.app.get("/api/mongodb-service/")
        async def check_mongodb_service():
            print("MongoDB Service is running")
            return JSONResponse(content={"message": "MongoDB Service is running"}, status_code=200)

        # -------------------------
        # Bucket Collection Endpoints
        # -------------------------
        @self.app.post("/api/mongodb-service/buckets/add-bucket")
        async def add_bucket(
            bucket_name: str,
        ):
            # create a new bucket with a generated bucketId
            if not bucket_name or not bucket_name.strip():
                raise HTTPException(status_code=400, detail="bucket_name is required")

            bucket_id = str(uuid.uuid4())
            doc = {"bucketId": bucket_id, "bucketName": bucket_name}
            try:
                self.buckets_collection.insert_one(doc)
            except DuplicateKeyError:
                raise HTTPException(status_code=409, detail="Bucket with this id already exists")
            return JSONResponse(status_code=201, content={"message": "Bucket created", "bucket": doc})

        @self.app.delete("/api/mongodb-service/buckets/delete-bucket")
        async def delete_bucket(
            bucket_id: str,
        ):
            if not bucket_id:
                raise HTTPException(status_code=400, detail="bucket_id is required")

            result = self.buckets_collection.delete_one({"bucketId": bucket_id})
            if result.deleted_count == 0:
                raise HTTPException(status_code=404, detail="Bucket not found")

            # Delete all leads associated with this bucket
            leads_result = self.leads_collection.delete_many({"bucketId": bucket_id})

            return JSONResponse(status_code=200, content={
                "message": "Bucket deleted",
                "bucketId": bucket_id,
                "deleted_bucket_count": result.deleted_count,
                "deleted_leads_count": leads_result.deleted_count,
            })

        @self.app.put("/api/mongodb-service/buckets/update-bucket-name")
        async def update_bucket_name(
            bucket_id: str,
            bucket_name: str,
        ):
            if not bucket_id or not bucket_name:
                raise HTTPException(status_code=400, detail="bucket_id and bucket_name are required")

            result = self.buckets_collection.update_one({"bucketId": bucket_id}, {"$set": {"bucketName": bucket_name}})
            if result.matched_count == 0:
                raise HTTPException(status_code=404, detail="Bucket not found")

            return JSONResponse(status_code=200, content={"message": "Bucket name updated", "bucketId": bucket_id, "bucketName": bucket_name})

        # -------------------------
        # Leads Collection Endpoints
        # -------------------------
        @self.app.get("/api/mongodb-service/leads/get-all-leads")
        async def get_all_leads(
            bucket_id: str = None,
        ):
            query = {}
            if bucket_id:
                query["bucketId"] = bucket_id

            cursor = self.leads_collection.find(query)
            leads = []
            # pymongo returns a sync cursor; iterate normally
            for doc in cursor:
                lead = dict(doc)
                if "_id" in lead:
                    lead.pop("_id")
                if "createdAt" in lead and isinstance(lead["createdAt"], datetime):
                    lead["createdAt"] = lead["createdAt"].isoformat()
                leads.append(lead)

            return JSONResponse(status_code=200, content={"leads": leads})

        @self.app.post("/api/mongodb-service/leads/add-lead")
        async def add_lead(
            url: str,
            username: str = None,
            platform: str = None,
            status: str = "new",
            bucket_id: str = None,
            notes : str = "",
        ):
            # basic validation
            if not url:
                raise HTTPException(status_code=400, detail="url is required")

            if not bucket_id:
                raise HTTPException(status_code=400, detail="bucket_id is required and must reference an existing bucket")

            # normalize required fields to comply with schema
            username = username or ""
            platform = platform or ""

            # verify bucket exists
            if self.buckets_collection.count_documents({"bucketId": bucket_id}) == 0:
                raise HTTPException(status_code=400, detail="Referenced bucket_id does not exist")

            # validate status
            allowed_status = {"new", "contacted", "converted", "closed"}
            if status not in allowed_status:
                raise HTTPException(status_code=400, detail=f"status must be one of: {', '.join(allowed_status)}")

            lead_id = str(uuid.uuid4())
            created_at = datetime.utcnow()
            doc = {
                "leadId": lead_id,
                "url": url,
                "username": username,
                "platform": platform,
                "status": status,
                "bucketId": bucket_id,
                "notes": notes if notes != "" else None,
                "createdAt": created_at,
            }
            try:
                self.leads_collection.insert_one(doc)
            except DuplicateKeyError:
                raise HTTPException(status_code=409, detail="Lead with this id already exists")

            # convert createdAt for response
            resp = dict(doc)
            resp["createdAt"] = created_at.isoformat()
            return JSONResponse(status_code=201, content={"message": "Lead created", "lead": resp})

        @self.app.put("/api/mongodb-service/leads/update-lead-status")
        async def update_lead_status(
            lead_id: str,
            status: str,
        ):
            if not lead_id or not status:
                raise HTTPException(status_code=400, detail="lead_id and status are required")

            allowed_status = {"new", "contacted", "converted", "closed"}
            if status not in allowed_status:
                raise HTTPException(status_code=400, detail=f"status must be one of: {', '.join(allowed_status)}")

            result = self.leads_collection.update_one({"leadId": lead_id}, {"$set": {"status": status}})
            if result.matched_count == 0:
                raise HTTPException(status_code=404, detail="Lead not found")

            return JSONResponse(status_code=200, content={"message": "Lead status updated", "leadId": lead_id, "status": status})

        @self.app.put("/api/mongodb-service/leads/update-lead-notes")
        async def update_lead_notes(
            lead_id: str,
            notes: str,
        ):
            if not lead_id:
                raise HTTPException(status_code=400, detail="lead_id is required")

            # Allow notes to be empty string -> store as None
            notes_val = notes if notes != "" else None

            result = self.leads_collection.update_one({"leadId": lead_id}, {"$set": {"notes": notes_val}})
            if result.matched_count == 0:
                raise HTTPException(status_code=404, detail="Lead not found")

            return JSONResponse(status_code=200, content={"message": "Lead notes updated", "leadId": lead_id, "notes": notes_val})
       
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