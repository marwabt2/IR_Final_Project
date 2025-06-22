import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

import joblib
from rank_bm25 import BM25Okapi
from fastapi import FastAPI
from backend.database.connection import get_mongo_connection
from backend.services.text_processing_service import processed_text
from backend.logger_config import logger

def create_bm25_representation(dataset_path: str):
    logger.info('Start creating BM25 representation')

    db = get_mongo_connection()
    collection_name = dataset_path.replace("/", "_")
    collection = db[collection_name]

    all_doc_ids = []
    all_tokenized_texts = []

    cursor = collection.find({}, {"_id": 0, "doc_id": 1, "text": 1})
    for doc in cursor:
        if "doc_id" in doc and "text" in doc:
            all_doc_ids.append(doc["doc_id"])
            tokens = processed_text(doc["text"]).split()
            all_tokenized_texts.append(tokens)

    logger.info(f"Training BM25 on {len(all_tokenized_texts)} documents")
    bm25 = BM25Okapi(all_tokenized_texts)

    safe_name = dataset_path.replace("/", "__")
    db_dir = os.path.join("db", safe_name)
    os.makedirs(db_dir, exist_ok=True)

    joblib.dump(bm25, os.path.join(db_dir, "bm25_model.joblib"), compress=3)
    joblib.dump(all_doc_ids, os.path.join(db_dir, "doc_ids.joblib"))
    joblib.dump(all_tokenized_texts, os.path.join(db_dir, "all_tokenized_texts.joblib"))

    logger.info("BM25 representation created and saved for %s", dataset_path)

if __name__ == "__main__":
    create_bm25_representation("vaswani")
