import os
import joblib
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.text_processing_service import processed_text
from backend.database.connection import get_mongo_connection
from backend.logger_config import logger
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter()

class TFIDFSearchRequest(BaseModel):
    query: str
    dataset_path: str
    top_k: int = 50

def load_tfidf_components(dataset_path: str):
    safe_name = dataset_path.replace("/", "__")
    base_path = os.path.join("db", safe_name)
    tfidf_matrix = joblib.load(os.path.join(base_path, "tfidf_matrix.joblib"))
    vectorizer = joblib.load(os.path.join(base_path, "vectorizer.joblib"))
    doc_ids = joblib.load(os.path.join(base_path, "doc_ids.joblib"))
    return tfidf_matrix, vectorizer, doc_ids

def load_documents_by_ids(dataset_path: str, doc_ids):
    db = get_mongo_connection()
    collection = db[dataset_path.replace("/", "_")]
    docs = {}
    for doc_id in doc_ids:
        doc = collection.find_one({"doc_id": doc_id}, {"_id": 0, "text": 1})
        if doc:
            docs[doc_id] = doc["text"]
    return docs

@router.post("/tfidf/search")
def tfidf_search(request: TFIDFSearchRequest):
    query = request.query
    dataset_path = request.dataset_path
    top_k = request.top_k

    logger.info(f"TF-IDF Weighted Index Search on dataset: {dataset_path}")

    tfidf_matrix, vectorizer, doc_ids = load_tfidf_components(dataset_path)
    processed_query = processed_text(query)
    query_vector = vectorizer.transform([processed_query])

    safe_name = dataset_path.replace("/", "__")
    weighted_index_path = os.path.join("db", safe_name, "weighted_inverted_index.joblib")
    if not os.path.exists(weighted_index_path):
        return {"error": f"Weighted inverted index not found for dataset: {dataset_path}"}

    weighted_index = joblib.load(weighted_index_path)

    candidate_doc_ids = set()
    for term in processed_query.split():
        if term in weighted_index:
            candidate_doc_ids.update([entry["doc_id"] for entry in weighted_index[term]])

    if not candidate_doc_ids:
        logger.warning("No matching documents found in weighted inverted index.")
        return {"results": []}

    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    candidate_indices = [doc_id_to_index[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_index]

    if not candidate_indices:
        logger.warning("No candidate document indices found in tfidf matrix.")
        return {"results": []}

    candidate_matrix = tfidf_matrix[candidate_indices]
    similarities = cosine_similarity(query_vector, candidate_matrix).flatten()

    sorted_indices = np.argsort(similarities)[::-1][:top_k]
    top_candidate_indices = [candidate_indices[i] for i in sorted_indices]
    top_doc_ids = [doc_ids[i] for i in top_candidate_indices]

    doc_texts = load_documents_by_ids(dataset_path, top_doc_ids)

    results = []
    for i, doc_id in enumerate(top_doc_ids):
        results.append({
            "doc_id": doc_id,
            "score": float(similarities[sorted_indices[i]]),
            "text": doc_texts.get(doc_id, "")
        })

    return {"results": results}
