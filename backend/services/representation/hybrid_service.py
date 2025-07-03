import os
import joblib
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import faiss

from backend.logger_config import logger

router = APIRouter()

class BuildHybridRequest(BaseModel):
    dataset_path: str
    tfidf_weight: float = 0.4
    bert_weight: float = 0.6
    tfidf_components: int = 300

@router.post("/hybrid/build")
def build_hybrid_representation(request: BuildHybridRequest):
    dataset_path = request.dataset_path
    tfidf_weight = request.tfidf_weight
    bert_weight = request.bert_weight
    n_components = request.tfidf_components

    safe_name = dataset_path.replace("/", "__")
    db_dir = os.path.join("db", safe_name)

    logger.info(f"Start building hybrid representation for dataset: {dataset_path}")

    tfidf_matrix = joblib.load(os.path.join(db_dir, "tfidf_matrix.joblib"))  # sparse
    docs_df = joblib.load(os.path.join(db_dir, "docs.joblib"))
    bert_embeddings = joblib.load(os.path.join(db_dir, "bert_embeddings.joblib"))  # dense
    bert_doc_ids = joblib.load(os.path.join(db_dir, "bert_doc_ids.joblib"))

    if tfidf_matrix.shape[0] != len(bert_embeddings):
        return {"error": f"Document count mismatch: TF-IDF={tfidf_matrix.shape[0]}, BERT={len(bert_embeddings)}"}

    logger.info("Reducing TF-IDF dimensions via TruncatedSVD...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)
    joblib.dump(svd, os.path.join(db_dir, "svd_model.joblib"))

    if tfidf_reduced.shape != bert_embeddings.shape:
        bert_embeddings = bert_embeddings[:, :tfidf_reduced.shape[1]]

    hybrid_representations = tfidf_weight * tfidf_reduced + bert_weight * bert_embeddings

    joblib.dump(hybrid_representations, os.path.join(db_dir, "hybrid_matrix.joblib"))
    joblib.dump(docs_df["pid"].tolist(), os.path.join(db_dir, "hybrid_doc_ids.joblib"))

    # ✅ Build FAISS index directly from hybrid vectors
    logger.info("Building FAISS index from hybrid vectors...")
    hybrid_matrix_norm = normalize(hybrid_representations, norm='l2', axis=1)
    index = faiss.IndexFlatIP(hybrid_matrix_norm.shape[1])
    index.add(hybrid_matrix_norm.astype(np.float32))
    faiss_index_path = os.path.join(db_dir, "hybrid_faiss.index")
    faiss.write_index(index, faiss_index_path)

    logger.info(f"Hybrid FAISS index created at: {faiss_index_path}")
    return {
        "status": "✅ Hybrid representation + FAISS built successfully",
        "documents": len(hybrid_representations),
        "dimension": hybrid_representations.shape[1]
    }
