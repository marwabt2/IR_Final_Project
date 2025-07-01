import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

import joblib
from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.text_processing_service import bm25_processed_text
from backend.database.connection import get_mongo_connection
from backend.logger_config import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from nltk.corpus import wordnet

router = APIRouter()

class BM25SearchRequest(BaseModel):
    query: str
    dataset_path: str
    top_k: int = 10
    initial_k: int = 50  # نستخدمه داخلياً قبل إعادة الترتيب

def expand_query(tokens):
    expanded = set(tokens)
    for token in tokens:
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))
    return list(expanded)

def load_bm25_components(dataset_path: str):
    safe_name = dataset_path.replace("/", "__")
    base_path = os.path.join("db", safe_name)
    bm25_model = joblib.load(os.path.join(base_path, "bm25_model.joblib"))
    doc_ids = joblib.load(os.path.join(base_path, "doc_ids.joblib"))
    tokenized_texts = joblib.load(os.path.join(base_path, "all_tokenized_texts.joblib"))
    return bm25_model, doc_ids, tokenized_texts

def load_documents_by_ids(dataset_path: str, doc_ids):
    db = get_mongo_connection()
    collection = db[dataset_path.replace("/", "_")]
    docs = {}
    for doc_id in doc_ids:
        doc = collection.find_one({"doc_id": doc_id}, {"_id": 0, "text": 1})
        if doc:
            docs[doc_id] = doc["text"]
    return docs

@router.post("/bm25/search")
def bm25_search(request: BM25SearchRequest):
    query = request.query
    dataset_path = request.dataset_path
    top_k = request.top_k
    initial_k = request.initial_k

    logger.info(f"BM25 + Cross-Encoder Re-ranking on dataset: {dataset_path}")

    bm25_model, doc_ids, tokenized_texts = load_bm25_components(dataset_path)

    query_tokens = bm25_processed_text(query)
    if not query_tokens:
        logger.warning("Query processing resulted in no tokens.")
        return []

    expanded_query = expand_query(query_tokens)

    safe_name = dataset_path.replace("/", "__")
    weighted_index_path = os.path.join("db", safe_name, "weighted_inverted_index.joblib")
    if not os.path.exists(weighted_index_path):
        raise FileNotFoundError(f"Weighted inverted index not found for dataset: {dataset_path}")

    weighted_index = joblib.load(weighted_index_path)

    candidate_doc_ids = set()
    for term in expanded_query:
        if term in weighted_index:
            candidate_doc_ids.update([entry["doc_id"] for entry in weighted_index[term]])

    if not candidate_doc_ids:
        logger.warning("No matching documents found in weighted inverted index.")
        return []

    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    candidate_indices = [doc_id_to_index[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_index]

    if not candidate_indices:
        logger.warning("No candidate indices matched in BM25 model.")
        return []

    scores = bm25_model.get_scores(expanded_query)
    candidate_scores = [(i, scores[i]) for i in candidate_indices]
    candidate_scores.sort(key=lambda x: x[1], reverse=True)

    top_candidates = candidate_scores[:initial_k]
    top_indices = [i for i, _ in top_candidates]
    top_doc_ids = [doc_ids[i] for i in top_indices]
    doc_texts = load_documents_by_ids(dataset_path, top_doc_ids)

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    cross_inputs = [(query, doc_texts[doc_ids[i]]) for i in top_indices]
    rerank_scores = cross_encoder.predict(cross_inputs)

    reranked = sorted(zip(top_doc_ids, rerank_scores), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for doc_id, score in reranked:
        results.append({
            "doc_id": doc_id,
            "score": float(score),
            "text": doc_texts[doc_id]
        })

    return results
