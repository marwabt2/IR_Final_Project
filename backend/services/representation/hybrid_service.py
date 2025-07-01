import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

import numpy as np
import joblib
import faiss
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.preprocessing import normalize
from sentence_transformers.cross_encoder import CrossEncoder

from backend.services.text_processing_service import processed_text
from backend.services.representation.bert_service import embed_text
from backend.database.connection import get_mongo_connection
from backend.logger_config import logger

router = APIRouter()

class HybridSearchRequest(BaseModel):
    query: str
    dataset_path: str
    alpha: float = 0.3
    top_k: int = 30
    rerank_k: int = 500

def load_tfidf_components(dataset_path):
    safe_name = dataset_path.replace("/", "__")
    base_path = os.path.join("db", safe_name)
    vectorizer = joblib.load(os.path.join(base_path, "vectorizer.joblib"))
    tfidf_doc_ids = joblib.load(os.path.join(base_path, "doc_ids.joblib"))
    return vectorizer, tfidf_doc_ids

def load_bert_doc_ids(dataset_path):
    safe_name = dataset_path.replace("/", "__")
    base_path = os.path.join("db", safe_name)
    bert_doc_ids = joblib.load(os.path.join(base_path, "bert_doc_ids.joblib"))
    return bert_doc_ids

def load_weighted_index(dataset_path):
    safe_name = dataset_path.replace("/", "__")
    path = os.path.join("db", safe_name, "weighted_inverted_index.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def faiss_search(query_vector, index_path, top_k):
    index = faiss.read_index(index_path)
    query_vector = query_vector.astype(np.float32).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    return indices[0], distances[0]

def load_documents_by_ids(dataset_path, doc_ids):
    db = get_mongo_connection()
    collection = db[dataset_path.replace("/", "_")]
    docs = {}
    for doc_id in doc_ids:
        doc = collection.find_one({"doc_id": doc_id}, {"_id": 0, "text": 1})
        if doc:
            docs[doc_id] = doc["text"]
    return docs

@router.post("/hybrid/search")
def hybrid_search_api(request: HybridSearchRequest):
    query = request.query
    dataset_path = request.dataset_path
    alpha = request.alpha
    top_k = request.top_k
    rerank_k = request.rerank_k

    logger.info(f"Hybrid + FAISS + Cross-Encoder Rerank Search on {dataset_path}")

    vectorizer, tfidf_doc_ids = load_tfidf_components(dataset_path)
    bert_doc_ids = load_bert_doc_ids(dataset_path)

    if tfidf_doc_ids != bert_doc_ids:
        return {"error": "Mismatch between TF-IDF and BERT doc IDs"}

    processed_query = processed_text(query)
    query_tfidf_vector = vectorizer.transform([processed_query])
    query_bert_vector = embed_text(processed_query).reshape(1, -1)

    query_tfidf_norm = normalize(query_tfidf_vector, norm='l2', axis=1).toarray()
    query_bert_norm = normalize(query_bert_vector, norm='l2', axis=1)

    hybrid_query = np.concatenate([alpha * query_tfidf_norm, (1 - alpha) * query_bert_norm], axis=1)

    safe_name = dataset_path.replace("/", "__")
    faiss_index_path = os.path.join("db", safe_name, "hybrid_faiss.index")

    weighted_index = load_weighted_index(dataset_path)
    if weighted_index is None:
        return {"error": "Weighted inverted index not found"}

    candidate_doc_ids = set()
    for term in processed_query.split():
        if term in weighted_index:
            candidate_doc_ids.update(weighted_index[term].keys())

    if not candidate_doc_ids:
        return {"results": []}

    top_indices, hybrid_sim_scores = faiss_search(hybrid_query, faiss_index_path, rerank_k)
    all_faiss_doc_ids = [tfidf_doc_ids[i] for i in top_indices]

    filtered_doc_ids = [doc_id for doc_id in all_faiss_doc_ids if doc_id in candidate_doc_ids]
    filtered_scores = [score for doc_id, score in zip(all_faiss_doc_ids, hybrid_sim_scores) if doc_id in candidate_doc_ids]

    if not filtered_doc_ids:
        return {"results": []}

    doc_texts = load_documents_by_ids(dataset_path, filtered_doc_ids)

    model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    local_dir = 'models/msmarco_cross_encoder'

    if not os.path.exists(local_dir):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)

    model = CrossEncoder(local_dir)

    pairs = [(query, doc_texts[doc_id]) for doc_id in filtered_doc_ids]
    rerank_scores = model.predict(pairs)

    rerank_dict = dict(zip(filtered_doc_ids, rerank_scores))
    hybrid_dict = dict(zip(filtered_doc_ids, filtered_scores))

    final_ranking = sorted(
        filtered_doc_ids,
        key=lambda doc_id: 0.6 * rerank_dict[doc_id] + 0.4 * hybrid_dict[doc_id],
        reverse=True
    )

    results = []
    for doc_id in final_ranking[:top_k]:
        results.append({
            "doc_id": doc_id,
            "score": float(rerank_dict[doc_id]),
            "text": doc_texts[doc_id]
        })

    return {"results": results}
