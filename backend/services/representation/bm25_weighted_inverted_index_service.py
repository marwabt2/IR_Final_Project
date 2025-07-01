import os
import joblib
from fastapi import APIRouter
from pydantic import BaseModel
from collections import defaultdict

from backend.database.connection import get_mongo_connection
from backend.logger_config import logger
from backend.services.text_processing_service import bm25_processed_text

router = APIRouter()

class WeightedIndexRequest(BaseModel):
    dataset_path: str

@router.post("/weighted_index/create_bm25")
def create_bm25_weighted_inverted_index(request: WeightedIndexRequest):
    dataset_path = request.dataset_path
    db = get_mongo_connection()
    collection = db[dataset_path.replace("/", "_")]

    inverted_index = defaultdict(list)

    logger.info(f"🔄 Creating BM25-style weighted inverted index for dataset: {dataset_path}")

    cursor = collection.find({}, {"_id": 0, "doc_id": 1, "text": 1})
    total_docs = 0

    for doc in cursor:
        if "doc_id" in doc and "text" in doc:
            total_docs += 1
            doc_id = doc["doc_id"]
            tokens = bm25_processed_text(doc["text"])
            token_freq = defaultdict(int)
            for token in tokens:
                token_freq[token] += 1
            for token, freq in token_freq.items():
                inverted_index[token].append({
                    "doc_id": doc_id,
                    "weight": freq
                })

    safe_name = dataset_path.replace("/", "__")
    os.makedirs(os.path.join("db", safe_name), exist_ok=True)
    output_path = os.path.join("db", safe_name, "bm25_weighted_inverted_index.joblib")
    joblib.dump(dict(inverted_index), output_path)

    logger.info(f"✅ BM25-style weighted inverted index created and saved at: {output_path}")
    return {
        "status": "BM25 weighted inverted index created successfully",
        "terms_count": len(inverted_index),
        "documents_indexed": total_docs
    }
