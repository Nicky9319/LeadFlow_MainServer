import json
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import CollectionInvalid

def create_mongo_db():
    # Load the MongoDB schema
    schema_path = os.path.join(os.path.dirname(__file__), "./mongo-db-schema.json")
    with open(schema_path, 'r') as f:
        schema_data = json.load(f)

    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/', server_api=ServerApi('1'))

    # Get database name from schema
    db_name = schema_data["database"]
    
    # Drop database if it exists for a clean start
    client.drop_database(db_name)
    print(f"Creating database: {db_name}")
    
    db = client[db_name]

    # Create collections with validators
    for collection_info in schema_data["collections"]:
        collection_name = collection_info["name"]
        
        try:
            # Create collection with validator
            db.create_collection(
                collection_name, 
                validator=collection_info["validator"],
                validationLevel=collection_info.get("validationLevel", "strict"),
                validationAction=collection_info.get("validationAction", "error")
            )
            
            print(f"Created collection: {collection_name}")
        except CollectionInvalid as e:
            print(f"Error creating collection {collection_name}: {e}")

    print(f"Successfully created MongoDB database '{db_name}' with all collections.")
    return True

if __name__ == "__main__":
    create_mongo_db()