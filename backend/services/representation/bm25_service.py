import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

import joblib
from rank_bm25 import BM25Okapi
from fastapi import FastAPI
from backend.database.connection import get_mongo_connection
from backend.services.text_processing_service import bm25_processed_text  # النسخة المحسنة
from backend.logger_config import logger
from fastapi import APIRouter
from pydantic import BaseModel

from tqdm import tqdm
import time

for i in tqdm(range(10)):
    time.sleep(0.5)

router = APIRouter()

class BM25Request(BaseModel):
    dataset_path: str

@router.post("/bm25/build")
def create_bm25_representation(request:BM25Request):
    logger.info('Start creating BM25 representation')

    db = get_mongo_connection()
    collection_name = request.dataset_path.replace("/", "_")
    collection = db[collection_name]

    all_doc_ids = []
    all_tokenized_texts = []

    cursor = collection.find({}, {"_id": 0, "doc_id": 1, "text": 1})
    for doc in tqdm(cursor, desc="Processing documents"):
        if "doc_id" in doc and "text" in doc:
            tokens = bm25_processed_text(doc["text"])
            if tokens:  # تأكد من عدم وجود وثائق فارغة
                all_doc_ids.append(doc["doc_id"])
                all_tokenized_texts.append(tokens)

    logger.info(f"Training BM25 on {len(all_tokenized_texts)} documents")
    # ضبط k1 و b لرفع MAP
    bm25 = BM25Okapi(all_tokenized_texts, k1=1.6, b=0.75)

    safe_name = request.dataset_path.replace("/", "__")
    db_dir = os.path.join("db", safe_name)
    os.makedirs(db_dir, exist_ok=True)

    joblib.dump(bm25, os.path.join(db_dir, "bm25_model.joblib"), compress=3)
    joblib.dump(all_doc_ids, os.path.join(db_dir, "doc_ids.joblib"))
    joblib.dump(all_tokenized_texts, os.path.join(db_dir, "all_tokenized_texts.joblib"))

    logger.info("BM25 representation created and saved for %s", request.dataset_path)
    return {
        "status": "BM25 created successfully",
        "documents_processed": len(all_doc_ids),
    }

if __name__ == "__main__":
    create_bm25_representation("vaswani")
