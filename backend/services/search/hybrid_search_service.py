# import os
# import sys
# sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

# import numpy as np
# import joblib
# import faiss
# from fastapi import APIRouter
# from pydantic import BaseModel
# from sklearn.preprocessing import normalize
# from sentence_transformers.cross_encoder import CrossEncoder

# from backend.services.text_processing_service import processed_text
# from backend.services.representation.bert_service import embed_text
# from backend.database.connection import get_mongo_connection
# from backend.logger_config import logger

# router = APIRouter()

# class HybridSearchRequest(BaseModel):
#     query: str
#     dataset_path: str
#     alpha: float = 0.3
#     top_k: int = 100
#     rerank_k: int = 500

# def load_tfidf_components(dataset_path):
#     safe_name = dataset_path.replace("/", "__")
#     base_path = os.path.join("db", safe_name)
#     vectorizer = joblib.load(os.path.join(base_path, "vectorizer.joblib"))
#     tfidf_doc_ids = joblib.load(os.path.join(base_path, "doc_ids.joblib"))
#     return vectorizer, tfidf_doc_ids

# def load_bert_doc_ids(dataset_path):
#     safe_name = dataset_path.replace("/", "__")
#     base_path = os.path.join("db", safe_name)
#     bert_doc_ids = joblib.load(os.path.join(base_path, "bert_doc_ids.joblib"))
#     return bert_doc_ids

# def faiss_search(query_vector, index_path, top_k):
#     index = faiss.read_index(index_path)
#     query_vector = query_vector.astype(np.float32).reshape(1, -1)
#     distances, indices = index.search(query_vector, top_k)
#     return indices[0], distances[0]

# def load_documents_by_ids(dataset_path, doc_ids):
#     db = get_mongo_connection()
#     collection = db[dataset_path.replace("/", "_")]
#     docs = {}
#     for doc_id in doc_ids:
#         doc = collection.find_one({"doc_id": doc_id}, {"_id": 0, "text": 1})
#         if doc:
#             docs[doc_id] = doc["text"]
#     return docs

# @router.post("/hybrid/search")
# def hybrid_search_api(request: HybridSearchRequest):
#     query = request.query
#     dataset_path = request.dataset_path
#     alpha = request.alpha
#     top_k = request.top_k
#     rerank_k = request.rerank_k

#     logger.info(f"Hybrid + FAISS + Cross-Encoder Rerank Search on {dataset_path}")

#     vectorizer, tfidf_doc_ids = load_tfidf_components(dataset_path)
#     bert_doc_ids = load_bert_doc_ids(dataset_path)

#     if tfidf_doc_ids != bert_doc_ids:
#         return {"error": "Mismatch between TF-IDF and BERT doc IDs"}

#     processed_query = processed_text(query)
#     query_tfidf_vector = vectorizer.transform([processed_query])
#     query_bert_vector = embed_text(processed_query).reshape(1, -1)

#     query_tfidf_norm = normalize(query_tfidf_vector, norm='l2', axis=1).toarray()
#     query_bert_norm = normalize(query_bert_vector, norm='l2', axis=1)

#     hybrid_query = np.concatenate([alpha * query_tfidf_norm, (1 - alpha) * query_bert_norm], axis=1)

#     safe_name = dataset_path.replace("/", "__")
#     faiss_index_path = os.path.join("db", safe_name, "hybrid_faiss.index")

#     top_indices, hybrid_sim_scores = faiss_search(hybrid_query, faiss_index_path, rerank_k)
#     candidate_ids = [tfidf_doc_ids[i] for i in top_indices]
#     doc_texts = load_documents_by_ids(dataset_path, candidate_ids)

#     model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
#     local_dir = 'models/msmarco_cross_encoder'

#     if not os.path.exists(local_dir):
#         from huggingface_hub import snapshot_download
#         snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)

#     model = CrossEncoder(local_dir)

#     pairs = [(query, doc_texts[doc_id]) for doc_id in candidate_ids]
#     rerank_scores = model.predict(pairs)

#     rerank_dict = dict(zip(candidate_ids, rerank_scores))
#     hybrid_dict = dict(zip(candidate_ids, hybrid_sim_scores))

#     final_ranking = sorted(
#         candidate_ids,
#         key=lambda doc_id: 0.6 * rerank_dict[doc_id] + 0.4 * hybrid_dict[doc_id],
#         reverse=True
#     )

#     results = []
#     for doc_id in final_ranking[:top_k]:
#         results.append({
#             "doc_id": doc_id,
#             "score": float(rerank_dict[doc_id]),
#             "text": doc_texts[doc_id]
#         })

#     return {"results": results}


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

from backend.services.text_processing_service import processed_text,TextProcessor
from backend.services.representation.bert_service import embed_text
from backend.database.connection import get_mongo_connection
from backend.logger_config import logger

router = APIRouter()

# كاش ثابت
vectorizer_cache = {}
doc_ids_cache = {}
faiss_index_cache = {}
bert_doc_ids_cache = {}

# تحميل CrossEncoder مرة واحدة
cross_encoder = CrossEncoder("models/msmarco_cross_encoder")


class HybridSearchRequest(BaseModel):
    query: str
    dataset_path: str
    alpha: float = 0.3
    top_k: int = 100
    rerank_k: int = 500


def get_vectorizer(dataset_path):
    if dataset_path not in vectorizer_cache:
        safe = dataset_path.replace("/", "__")
        path = os.path.join("db", safe, "vectorizer.joblib")
        vectorizer_cache[dataset_path] = joblib.load(path)
    return vectorizer_cache[dataset_path]


def get_doc_ids(dataset_path):
    if dataset_path not in doc_ids_cache:
        safe = dataset_path.replace("/", "__")
        path = os.path.join("db", safe, "doc_ids.joblib")
        doc_ids_cache[dataset_path] = joblib.load(path)
    return doc_ids_cache[dataset_path]


def get_bert_doc_ids(dataset_path):
    if dataset_path not in bert_doc_ids_cache:
        safe = dataset_path.replace("/", "__")
        path = os.path.join("db", safe, "bert_doc_ids.joblib")
        bert_doc_ids_cache[dataset_path] = joblib.load(path)
    return bert_doc_ids_cache[dataset_path]


def get_faiss_index(dataset_path):
    if dataset_path not in faiss_index_cache:
        safe = dataset_path.replace("/", "__")
        path = os.path.join("db", safe, "hybrid_faiss.index")
        index = faiss.read_index(path)
        faiss_index_cache[dataset_path] = index
    return faiss_index_cache[dataset_path]


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

    logger.info(f"Hybrid + FAISS + Cross-Encoder Search for {dataset_path}")
    processor = TextProcessor()
    vectorizer = get_vectorizer(dataset_path)
    tfidf_doc_ids = get_doc_ids(dataset_path)
    bert_doc_ids = get_bert_doc_ids(dataset_path)

    if tfidf_doc_ids != bert_doc_ids:
        return {"error": "Mismatch between TF-IDF and BERT doc IDs"}

    doc_id_to_index = {str(doc_id): i for i, doc_id in enumerate(tfidf_doc_ids)}

    processed_query = processed_text(query,processor)
    query_tfidf_vector = vectorizer.transform([processed_query])
    query_bert_vector = embed_text(processed_query).reshape(1, -1)

    query_tfidf_norm = normalize(query_tfidf_vector, norm='l2', axis=1).toarray()
    query_bert_norm = normalize(query_bert_vector, norm='l2', axis=1)

    hybrid_query = np.concatenate([alpha * query_tfidf_norm, (1 - alpha) * query_bert_norm], axis=1)

    index = get_faiss_index(dataset_path)
    hybrid_query = hybrid_query.astype(np.float32).reshape(1, -1)
    top_indices, hybrid_sim_scores = index.search(hybrid_query, rerank_k)

    candidate_ids = [tfidf_doc_ids[i] for i in top_indices]
    doc_texts = load_documents_by_ids(dataset_path, candidate_ids)

    pairs = [(query, doc_texts[doc_id]) for doc_id in candidate_ids]
    rerank_scores = cross_encoder.predict(pairs)

    rerank_dict = dict(zip(candidate_ids, rerank_scores))
    hybrid_dict = dict(zip(candidate_ids, hybrid_sim_scores))

    final_ranking = sorted(
        candidate_ids,
        key=lambda doc_id: 0.6 * rerank_dict[doc_id] + 0.4 * hybrid_dict[doc_id],
        reverse=True
    )

    top_documents = []
    cosine_similarities = []
    top_documents_indices = []

    for doc_id in final_ranking[:top_k]:
        text = doc_texts[doc_id]
        score = rerank_dict[doc_id]
        top_documents.append({
            "doc_id": doc_id,
            "score": float(score),
            "text": text
        })
        cosine_similarities.append(float(score))
        top_documents_indices.append(doc_id_to_index.get(str(doc_id), -1))

    return {
        "top_documents": top_documents,
        "cosine_similarities": cosine_similarities,
        "top_documents_indices": top_documents_indices
    }
