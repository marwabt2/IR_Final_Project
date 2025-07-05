import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

from fastapi import APIRouter
from pydantic import BaseModel
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from backend.services.text_processing_service import processed_text,TextProcessor
from backend.database.connection import get_mongo_connection
from backend.logger_config import logger
import pandas as pd

router = APIRouter()

from functools import lru_cache

_cached_datasets = {}
def load_dataset_from_cache(dataset_path: str):
    if dataset_path in _cached_datasets:
        return _cached_datasets[dataset_path]

    logger.info(f"Loading dataset into cache: {dataset_path}")
    db = get_mongo_connection()
    collection_name = dataset_path.replace("/", "_")
    collection = db[collection_name]

    pids = []
    texts = []
    cursor = collection.find({}, {"_id": 0, "doc_id": 1, "text": 1})
    for doc in cursor:
        if "doc_id" in doc and "text" in doc and isinstance(doc["text"], str):
            pids.append(doc["doc_id"])
            texts.append(doc["text"])

    df = pd.DataFrame({"pid": pids, "text": texts})
    df.dropna(subset=['text'], inplace=True)

    _cached_datasets[dataset_path] = df  # نخزن بالذاكرة
    return df

class SearchRequest(BaseModel):
    query: str
    dataset_path: str  # مجلد التخزين: مثل db/lotte_dev
    top_n: int = 10

@router.post("/tfidf/search")
def search_documents(request: SearchRequest):
    logger.info(f"TF-IDF Weighted Index Search on dataset: {request.dataset_path}")
    data = load_dataset_from_cache(request.dataset_path)
    processor = TextProcessor()

    try:
        safe_name = request.dataset_path.replace("/", "__")
        db_dir = os.path.join("db", safe_name)
        vectorizer = joblib.load(os.path.join(db_dir, "vectorizer.joblib"))
        tfidf_matrix = joblib.load(os.path.join(db_dir, "tfidf_matrix.joblib"))
    except Exception as e:
        return {"error": f"❌ Failed to load TF-IDF components: {e}"}

    processed_query = processed_text(request.query, processor)
    query_vector = vectorizer.transform([processed_query])
    cosine_similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()
    top_documents_indices = cosine_similarities.argsort()[-request.top_n:][::-1]
    top_documents = data.iloc[top_documents_indices]
    return {
    "top_documents": top_documents.to_dict(orient="records"),
    "cosine_similarities": cosine_similarities[top_documents_indices].tolist(),
    "top_documents_indices": top_documents_indices.tolist()
}

@router.post("/tfidf/search/clustering")
def search_documents(request: SearchRequest):
    logger.info(f"TF-IDF Weighted Index Search on dataset: {request.dataset_path}")
    data = load_dataset_from_cache(request.dataset_path)
    processor = TextProcessor()

    try:
        safe_name = request.dataset_path.replace("/", "__")
        db_dir = os.path.join("db", safe_name)

        # تحميل ملفات TF-IDF
        vectorizer = joblib.load(os.path.join(db_dir, "vectorizer.joblib"))
        tfidf_matrix = joblib.load(os.path.join(db_dir, "tfidf_matrix.joblib"))

        # تحميل ملفات clustering
        kmeans_model = joblib.load(os.path.join(db_dir, "kmeans_model.joblib"))
        svd_model = joblib.load(os.path.join(db_dir, "svd_model.joblib"))
        doc_clusters = joblib.load(os.path.join(db_dir, "doc_clusters.joblib"))  # قائمة [0, 1, 2, ...]

    except Exception as e:
        return {"error": f"❌ Failed to load required models: {e}"}

    # معالجة الاستعلام
    processed_query = processed_text(request.query, processor)
    query_vector = vectorizer.transform([processed_query])

    # تقليل أبعاد الاستعلام وتصنيفه
    reduced_query = svd_model.transform(query_vector)
    query_cluster = kmeans_model.predict(reduced_query)[0]

    # تحديد المستندات التي تنتمي لنفس العنقود
    doc_indices_in_same_cluster = [
        idx for idx, cluster in enumerate(doc_clusters) if cluster == query_cluster
    ]

    if not doc_indices_in_same_cluster:
        return {"warning": "⚠️ No documents found in the same cluster as the query."}

    # حساب cosine similarity فقط على هالمجموعة
    filtered_tfidf = tfidf_matrix[doc_indices_in_same_cluster]
    cosine_similarities = cosine_similarity(filtered_tfidf, query_vector).flatten()
    sorted_indices = cosine_similarities.argsort()[-request.top_n:][::-1]

    # نحصل على النتائج الحقيقية من DataFrame الأصلي
    top_indices_real = [doc_indices_in_same_cluster[i] for i in sorted_indices]
    top_documents = data.iloc[top_indices_real]

    return {
        "query_cluster": int(query_cluster),
        "top_documents": top_documents.to_dict(orient="records"),
        "cosine_similarities": cosine_similarities[sorted_indices].tolist(),
        "top_documents_indices": top_indices_real
    }
