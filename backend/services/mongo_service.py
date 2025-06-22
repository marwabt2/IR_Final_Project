from pymongo import MongoClient # type: ignore

def get_mongo_collection(dataset_name: str, collection_name: str):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["information_retrieval"]
    return db[f"{dataset_name}_{collection_name}"]
