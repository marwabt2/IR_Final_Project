import os
import joblib
import numpy as np
import faiss
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.preprocessing import normalize

from backend.logger_config import logger

router = APIRouter()

class HybridRequest(BaseModel):
    dataset_path: str
    alpha: float = 0.3

def load_tfidf_matrix_and_doc_ids(dataset_path: str):
    safe_name = dataset_path.replace("/", "__")
    base_path = os.path.join("db", safe_name)
    tfidf_matrix = joblib.load(os.path.join(base_path, "tfidf_matrix.joblib"))
    doc_ids = joblib.load(os.path.join(base_path, "doc_ids.joblib"))
    return tfidf_matrix, doc_ids

def load_bert_embeddings_and_doc_ids(dataset_path: str):
    safe_name = dataset_path.replace("/", "__")
    base_path = os.path.join("db", safe_name)
    emb_path = os.path.join(base_path, "bert_embeddings.joblib")
    ids_path = os.path.join(base_path, "bert_doc_ids.joblib")
    if not os.path.exists(emb_path) or not os.path.exists(ids_path):
        raise FileNotFoundError("BERT embeddings or doc_ids file not found")
    embeddings = joblib.load(emb_path)
    doc_ids = joblib.load(ids_path)
    return embeddings, doc_ids

@router.post("/hybrid/build")
def create_hybrid_representation(request: HybridRequest):
    dataset_path = request.dataset_path
    alpha = request.alpha
    logger.info(f"Start building hybrid FAISS index for dataset: {dataset_path}")

    try:
        tfidf_matrix, tfidf_doc_ids = load_tfidf_matrix_and_doc_ids(dataset_path)
        bert_embeddings, bert_doc_ids = load_bert_embeddings_and_doc_ids(dataset_path)

        if tfidf_doc_ids != bert_doc_ids:
            return {"error": "Mismatch between TF-IDF and BERT doc IDs"}

        n_docs = len(tfidf_doc_ids)
        tfidf_dim = tfidf_matrix.shape[1]
        bert_dim = bert_embeddings.shape[1]

        logger.info(f"TF-IDF shape: {tfidf_matrix.shape}, BERT shape: {bert_embeddings.shape}")

        hybrid_embeddings = np.zeros((n_docs, tfidf_dim + bert_dim), dtype=np.float32)
        bert_embeddings_norm = normalize(bert_embeddings, norm='l2')

        for i in range(n_docs):
            tfidf_vec = tfidf_matrix.getrow(i).toarray().reshape(-1)
            tfidf_vec_norm = tfidf_vec / np.linalg.norm(tfidf_vec) if np.linalg.norm(tfidf_vec) > 0 else tfidf_vec
            hybrid_vec = np.concatenate([alpha * tfidf_vec_norm, (1 - alpha) * bert_embeddings_norm[i]])
            hybrid_embeddings[i] = hybrid_vec
            if (i + 1) % 10000 == 0:
                logger.info(f"Processed {i + 1}/{n_docs} documents")

        faiss.normalize_L2(hybrid_embeddings)
        index = faiss.IndexFlatIP(hybrid_embeddings.shape[1])
        index.add(hybrid_embeddings.astype(np.float32))

        safe_name = dataset_path.replace("/", "__")
        out_path = os.path.join("db", safe_name, "hybrid_faiss.index")
        faiss.write_index(index, out_path)

        logger.info(f"Hybrid FAISS index saved to: {out_path}")

        return {
            "status": "Hybrid FAISS index created successfully",
            "documents_processed": n_docs,
            "index_path": out_path
        }

    except FileNotFoundError as e:
        return {"error": str(e)}
