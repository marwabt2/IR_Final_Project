import os
import joblib
import numpy as np
import faiss
from fastapi import APIRouter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder

from backend.services.text_processing_service import processed_text
from backend.database.connection import get_mongo_connection
from backend.logger_config import logger

router = APIRouter()

retrieval_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

class BertSearchRequest(BaseModel):
    query: str
    dataset_path: str
    top_k: int = 300
    rerank_k: int = 50

def load_faiss_index(db_dir):
    embeddings = joblib.load(os.path.join(db_dir, "bert_embeddings.joblib"))
    doc_ids = joblib.load(os.path.join(db_dir, "bert_doc_ids.joblib"))
    index = faiss.read_index(os.path.join(db_dir, "bert_faiss.index"))
    return index, embeddings, doc_ids

def load_documents_by_ids(doc_ids, dataset_path):
    db = get_mongo_connection()
    collection = db[dataset_path.replace("/", "_")]
    cursor = collection.find({"doc_id": {"$in": doc_ids}}, {"_id": 0, "doc_id": 1, "text": 1})
    doc_map = {doc["doc_id"]: doc for doc in cursor}
    return [doc_map[doc_id] for doc_id in doc_ids if doc_id in doc_map]

@router.post("/bert/search")
def bert_search(request: BertSearchRequest):
    query = request.query
    dataset_path = request.dataset_path
    top_k = request.top_k
    rerank_k = request.rerank_k

    logger.info(f"Start search with FAISS + Cross-Encoder for dataset: {dataset_path}")
    safe_name = dataset_path.replace("/", "__")
    db_dir = os.path.join("db", safe_name)

    try:
        index, _, doc_ids = load_faiss_index(db_dir)
    except Exception as e:
        return {"error": f"Failed to load FAISS index or embeddings: {str(e)}"}

    query_processed = processed_text(query)
    query_vec = retrieval_model.encode(query_processed, normalize_embeddings=True).astype(np.float32).reshape(1, -1)

    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, top_k)

    top_doc_ids = [doc_ids[i] for i in indices[0]]
    top_docs = load_documents_by_ids(top_doc_ids, dataset_path)

    pairs = [(query, doc['text']) for doc in top_docs]
    rerank_scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(top_docs, rerank_scores), key=lambda x: x[1], reverse=True)
    ranked_results = [{
        "doc_id": doc["doc_id"],
        "score": float(score),
        "text": doc["text"]
    } for doc, score in ranked[:rerank_k]]

    return {"results": ranked_results}
