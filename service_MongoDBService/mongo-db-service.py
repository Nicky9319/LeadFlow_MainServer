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
        
    async def configure_routes(self):
        
        @self.app.get("/api/mongodb-service/")
        async def check_mongodb_service():
            print("MongoDB Service is running")
            return JSONResponse(content={"message": "MongoDB Service is running"}, status_code=200)

        
       
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