# backend/database/connection.py

from pymongo import MongoClient # type: ignore

def get_mongo_connection():
    client = MongoClient("mongodb://localhost:27017")
    db = client["information_retrieval"]
    return db
