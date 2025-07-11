import os
import joblib
import numpy as np
import faiss
from fastapi import APIRouter
from pydantic import BaseModel
from functools import lru_cache

from backend.database.connection import get_mongo_connection
from backend.services.text_processing_service import processed_text, TextProcessor
from backend.logger_config import logger

from sentence_transformers import SentenceTransformer

router = APIRouter()
processor = TextProcessor()

# كاش تحميل المودل لأنه بياخد وقت
@lru_cache(maxsize=1)
def load_bert_model():
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# كاش تحميل الملفات الثقيلة حسب المسار
_loaded_cache = {}

def load_cached(path, loader=joblib.load):
    if path not in _loaded_cache:
        _loaded_cache[path] = loader(path)
    return _loaded_cache[path]

class HybridSearchRequest(BaseModel):
    dataset_path: str
    query: str
    top_n: int = 10
    tfidf_weight: float = 0.4
    bert_weight: float = 0.6
    tfidf_components: int = 300

@router.post("/hybrid/search")
def hybrid_search(request: HybridSearchRequest):
    logger.info(f"Hybrid Search [tfidf+bert] on dataset: {request.dataset_path}")
    safe_name = request.dataset_path.replace("/", "__")
    db_dir = os.path.join("db", safe_name)

    # تحميل تمثيلات TF-IDF وBERT (مع كاش)
    tfidf_vectorizer = load_cached(os.path.join(db_dir, "vectorizer.joblib"))
    tfidf_matrix = load_cached(os.path.join(db_dir, "tfidf_matrix.joblib"))
    docs_df = load_cached(os.path.join(db_dir, "docs.joblib"))
    bert_embeddings = load_cached(os.path.join(db_dir, "bert_embeddings.joblib"))
    bert_doc_ids = load_cached(os.path.join(db_dir, "bert_doc_ids.joblib"))
    svd = load_cached(os.path.join(db_dir, "svd_model.joblib"))

    # تجهيز الكويري
    query_processed = processed_text(request.query, processor)

    # تمثيل TF-IDF للكويري
    tfidf_q = tfidf_vectorizer.transform([query_processed])
    tfidf_q_reduced = svd.transform(tfidf_q).astype(np.float32)

    # تمثيل BERT للكويري
    model = load_bert_model()
    bert_q = model.encode([query_processed], normalize_embeddings=True).astype(np.float32)

    # تحميل FAISS index ومعرفة الأبعاد المتوقعة
    faiss_index_path = os.path.join(db_dir, "hybrid_faiss.index")
    index = faiss.read_index(faiss_index_path)
    target_dim = index.d

    # دالة لضبط الأبعاد: قص أو padding بالأصفار
    def adjust_dim(vec, target_dim):
        current_dim = vec.shape[1]
        if current_dim > target_dim:
            return vec[:, :target_dim]
        elif current_dim < target_dim:
            padding = np.zeros((vec.shape[0], target_dim - current_dim), dtype=vec.dtype)
            return np.hstack([vec, padding])
        else:
            return vec

    # ضبط التمثيلات لتطابق الفهرس
    tfidf_vec = adjust_dim(tfidf_q_reduced, target_dim)
    bert_vec = adjust_dim(bert_q, target_dim)

    # الدمج
    hybrid_query = request.tfidf_weight * tfidf_vec + request.bert_weight * bert_vec
    hybrid_query = np.ascontiguousarray(hybrid_query)
    faiss.normalize_L2(hybrid_query)

    # البحث باستخدام FAISS
    D, I = index.search(hybrid_query, request.top_n)

    # استخراج النتائج
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append({
            "doc_id": docs_df.iloc[idx]["pid"],
            "score": float(score),
            "text": docs_df.iloc[idx]["text"]
        })

    return {
        "query": request.query,
        "top_documents": results,
        "cosine_similarities": D[0].tolist(),
        "top_documents_indices": I[0].tolist()
    }
