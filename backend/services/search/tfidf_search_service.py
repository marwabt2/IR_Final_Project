import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

from fastapi import APIRouter
from pydantic import BaseModel
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from backend.services.text_processing_service import processed_text,TextProcessor
from backend.database.connection import get_mongo_connection
from backend.logger_config import logger
import pandas as pd

router = APIRouter()

from functools import lru_cache

_cached_datasets = {}
def load_dataset_from_cache(dataset_path: str):
    if dataset_path in _cached_datasets:
        return _cached_datasets[dataset_path]

    logger.info(f"Loading dataset into cache: {dataset_path}")
    db = get_mongo_connection()
    collection_name = dataset_path.replace("/", "_")
    collection = db[collection_name]

    pids = []
    texts = []
    cursor = collection.find({}, {"_id": 0, "doc_id": 1, "text": 1})
    for doc in cursor:
        if "doc_id" in doc and "text" in doc and isinstance(doc["text"], str):
            pids.append(doc["doc_id"])
            texts.append(doc["text"])

    df = pd.DataFrame({"pid": pids, "text": texts})
    df.dropna(subset=['text'], inplace=True)

    _cached_datasets[dataset_path] = df  # نخزن بالذاكرة
    return df

class SearchRequest(BaseModel):
    query: str
    dataset_path: str  # مجلد التخزين: مثل db/lotte_dev
    top_n: int = 10

@router.post("/tfidf/search")
def search_documents(request: SearchRequest):
    logger.info(f"TF-IDF Weighted Index Search on dataset: {request.dataset_path}")
    data = load_dataset_from_cache(request.dataset_path)
    processor = TextProcessor()

    try:
        safe_name = request.dataset_path.replace("/", "__")
        db_dir = os.path.join("db", safe_name)
        vectorizer = joblib.load(os.path.join(db_dir, "vectorizer.joblib"))
        tfidf_matrix = joblib.load(os.path.join(db_dir, "tfidf_matrix.joblib"))
    except Exception as e:
        return {"error": f"❌ Failed to load TF-IDF components: {e}"}

    processed_query = processed_text(request.query, processor)
    query_vector = vectorizer.transform([processed_query])
    cosine_similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()
    top_documents_indices = cosine_similarities.argsort()[-request.top_n:][::-1]
    top_documents = data.iloc[top_documents_indices]
    return {
    "top_documents": top_documents.to_dict(orient="records"),
    "cosine_similarities": cosine_similarities[top_documents_indices].tolist(),
    "top_documents_indices": top_documents_indices.tolist()
}