import os
import joblib
import numpy as np
import faiss
from fastapi import APIRouter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder

from backend.services.text_processing_service import processed_text,TextProcessor
from backend.database.connection import get_mongo_connection
from backend.logger_config import logger
from backend.services.search.query_enhancement_service import query_enhancer


router = APIRouter()

# تحميل الموديلات مرة واحدة
retrieval_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# كاش لفهارس FAISS
faiss_cache = {}

# طلب المستخدم
class BertSearchRequest(BaseModel):
    query: str
    dataset_path: str
    top_k: int = 50
    rerank_k: int = 10

# تحميل FAISS index والـ doc_ids من الكاش أو من الملفات
def get_faiss_index_and_doc_ids(dataset_path: str):
    if dataset_path in faiss_cache:
        return faiss_cache[dataset_path]

    db_dir = os.path.join("db", dataset_path.replace("/", "__"))
    index = faiss.read_index(os.path.join(db_dir, "bert_faiss.index"))
    doc_ids = joblib.load(os.path.join(db_dir, "bert_doc_ids.joblib"))

    faiss_cache[dataset_path] = (index, doc_ids)
    return index, doc_ids

# تحميل كل الوثائق كماب: doc_id → text
def load_documents_map(dataset_path):
    db = get_mongo_connection()
    collection = db[dataset_path.replace("/", "_")]
    cursor = collection.find({}, {"_id": 0, "doc_id": 1, "text": 1})
    doc_map = {}
    for doc in cursor:
        if "doc_id" in doc and "text" in doc:
            doc_map[str(doc["doc_id"])] = doc["text"]
    return doc_map

@router.post("/bert/search_query_ref")
def bert_search(request: BertSearchRequest):
    query = request.query
    dataset_path = request.dataset_path
    top_k = request.top_k
    rerank_k = request.rerank_k

    logger.info(f"Start BERT ref search for dataset: {dataset_path}")
    processor = TextProcessor()
    try:
        index, doc_ids = get_faiss_index_and_doc_ids(dataset_path)
    except Exception as e:
        return {"error": f"Failed to load FAISS index: {str(e)}"}

    doc_id_to_index = {str(doc_id): i for i, doc_id in enumerate(doc_ids)}
    doc_map = load_documents_map(dataset_path)

    query_processed = processed_text(query, processor)
    corrected = query_enhancer.correct_spelling(query_processed)
    grammar_corrected = query_enhancer.correct_grammar_conservatively(corrected)
    expanded_query = query_enhancer.expand_query_minimally(grammar_corrected)

    query_vec = retrieval_model.encode(expanded_query, normalize_embeddings=True).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)
    top_doc_ids = [doc_ids[i] for i in indices[0]]
    top_docs = [(str(doc_id), doc_map.get(str(doc_id), "")) for doc_id in top_doc_ids]

    filtered_docs = [(doc_id, text) for doc_id, text in top_docs if text.strip()]
    pairs = [(query, text) for _, text in filtered_docs]
    rerank_scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(filtered_docs, rerank_scores), key=lambda x: x[1], reverse=True)
    reranked = ranked[:rerank_k]

    top_documents = []
    cosine_similarities = []
    top_documents_indices = []

    for (doc_id, text), score in reranked:
        top_documents.append({
            "doc_id": doc_id,
            "score": float(score),
            "text": text
        })
        cosine_similarities.append(float(score))
        top_documents_indices.append(doc_id_to_index.get(doc_id, -1))

    return {
        "top_documents": top_documents,
        "cosine_similarities": cosine_similarities,
        "top_documents_indices": top_documents_indices
    }
