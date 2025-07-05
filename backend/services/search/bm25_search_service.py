from fastapi import APIRouter
from pydantic import BaseModel
import os
import joblib
from backend.services.text_processing_service import bm25_processed_text
from backend.database.connection import get_mongo_connection
from backend.logger_config import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from nltk.corpus import wordnet

router = APIRouter()

# كاش CrossEncoder ثابت للتحميل مرة واحدة
cross_encoder = None

# كاش weighted_index لكل dataset_path
weighted_index_cache = {}

# كاش BM25 components لكل dataset_path
bm25_cache = {}

def get_cross_encoder():
    global cross_encoder
    if cross_encoder is None:
        logger.info("Loading CrossEncoder model for the first time...")
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return cross_encoder

def get_weighted_index(dataset_path):
    safe_name = dataset_path.replace("/", "__")
    if safe_name not in weighted_index_cache:
        weighted_index_path = os.path.join("db", safe_name, "weighted_inverted_index.joblib")
        logger.info(f"Loading weighted inverted index from {weighted_index_path} ...")
        weighted_index_cache[safe_name] = joblib.load(weighted_index_path)
    return weighted_index_cache[safe_name]

def get_bm25_components(dataset_path):
    safe_name = dataset_path.replace("/", "__")
    if safe_name not in bm25_cache:
        base_path = os.path.join("db", safe_name)
        logger.info(f"Loading BM25 components from {base_path} ...")
        bm25_model = joblib.load(os.path.join(base_path, "bm25_model.joblib"))
        doc_ids = joblib.load(os.path.join(base_path, "doc_ids.joblib"))
        tokenized_texts = joblib.load(os.path.join(base_path, "all_tokenized_texts.joblib"))
        bm25_cache[safe_name] = (bm25_model, doc_ids, tokenized_texts)
    return bm25_cache[safe_name]

class BM25SearchRequest(BaseModel):
    query: str
    dataset_path: str
    top_k: int = 10
    initial_k: int = 30

def expand_query(tokens):
    expanded = set(tokens)
    for token in tokens:
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))
    return list(expanded)

def load_documents_by_ids(dataset_path: str, doc_ids):
    db = get_mongo_connection()
    collection = db[dataset_path.replace("/", "_")]
    cursor = collection.find({"doc_id": {"$in": list(doc_ids)}}, {"_id": 0, "doc_id": 1, "text": 1})
    return {doc["doc_id"]: doc["text"] for doc in cursor if "text" in doc}

@router.post("/bm25/search")
async def bm25_search_eval(request: BM25SearchRequest):
    query = request.query
    dataset_path = request.dataset_path
    top_k = request.top_k
    initial_k = request.initial_k

    logger.info(f"[BM25 Eval] BM25 + Cross-Encoder Re-ranking on dataset: {dataset_path}")

    bm25_model, doc_ids, tokenized_texts = get_bm25_components(dataset_path)

    query_tokens = bm25_processed_text(query)
    if not query_tokens:
        logger.warning("Query processing resulted in no tokens.")
        return {"top_documents": [], "cosine_similarities": [], "top_documents_indices": []}

    expanded_query = expand_query(query_tokens)

    weighted_index = get_weighted_index(dataset_path)

    candidate_doc_ids = set()
    for term in expanded_query:
        if term in weighted_index:
            for entry in weighted_index[term]:
                if isinstance(entry, dict):
                    candidate_doc_ids.add(entry["doc_id"])
                else:
                    candidate_doc_ids.add(entry)

    if not candidate_doc_ids:
        logger.warning("No matching documents found in weighted inverted index.")
        return {"top_documents": [], "cosine_similarities": [], "top_documents_indices": []}

    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    candidate_indices = [doc_id_to_index[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_index]

    if not candidate_indices:
        logger.warning("No candidate indices matched in BM25 model.")
        return {"top_documents": [], "cosine_similarities": [], "top_documents_indices": []}

    scores = bm25_model.get_scores(expanded_query)
    candidate_scores = [(i, scores[i]) for i in candidate_indices]
    candidate_scores.sort(key=lambda x: x[1], reverse=True)

    top_candidates = candidate_scores[:initial_k]
    top_indices = [i for i, _ in top_candidates]
    top_doc_ids = [doc_ids[i] for i in top_indices]
    doc_texts = load_documents_by_ids(dataset_path, top_doc_ids)

    cross_encoder = get_cross_encoder()

    cross_inputs = [(query, doc_texts[doc_ids[i]]) for i in top_indices]
    rerank_scores = cross_encoder.predict(cross_inputs, batch_size=16)

    reranked = sorted(zip(top_doc_ids, rerank_scores, top_indices), key=lambda x: x[1], reverse=True)[:top_k]

    top_documents = []
    cosine_similarities = []
    top_documents_indices = []

    for doc_id, score, idx in reranked:
        top_documents.append({
            "doc_id": doc_id,
            "score": float(score),
            "text": doc_texts[doc_id]
        })
        cosine_similarities.append(float(score))
        top_documents_indices.append(idx)

    return {
        "top_documents": top_documents,
        "cosine_similarities": cosine_similarities,
        "top_documents_indices": top_documents_indices
    }
