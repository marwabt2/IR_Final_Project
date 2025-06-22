import os
import joblib
from fastapi import APIRouter
from pydantic import BaseModel
from backend.database.connection import get_mongo_connection
from backend.logger_config import logger
from sklearn.feature_extraction.text import TfidfVectorizer

router = APIRouter()

class WeightedIndexRequest(BaseModel):
    dataset_path: str

def _load_joblib_data(dataset_path: str, filename: str):
    safe_name = dataset_path.replace("/", "__")
    path = os.path.join("db", safe_name, filename)
    if os.path.exists(path):
        return joblib.load(path)
    raise FileNotFoundError(f"{filename} not found for dataset {dataset_path}")

def load_vectorizer(dataset_path: str):
    safe_name = dataset_path.replace("/", "__")
    path = os.path.join("db", safe_name, "vectorizer.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    else:
        logger.warning(f"⚠️ Vectorizer not found for dataset path: {dataset_path}")
        return None

@router.post("/weighted_index/create")
def create_weighted_inverted_index(request: WeightedIndexRequest):
    dataset_path = request.dataset_path

    try:
        tfidf_matrix = _load_joblib_data(dataset_path, "tfidf_matrix.joblib")
        doc_ids = _load_joblib_data(dataset_path, "doc_ids.joblib")
        vectorizer = load_vectorizer(dataset_path)
        if vectorizer is None:
            return {"error": "Vectorizer not found, cannot create weighted inverted index."}

        terms = vectorizer.get_feature_names_out()
        weighted_index = {}

        logger.info("🔄 Creating weighted inverted index...")

        for doc_index in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[doc_index]
            doc_id = doc_ids[doc_index]
            for term_index in row.indices:
                term = terms[term_index]
                weight = round(float(row[0, term_index]), 4)
                weighted_index.setdefault(term, []).append({
                    "doc_id": doc_id,
                    "weight": weight
                })

        db = get_mongo_connection()
        collection_name = f"{dataset_path.replace('/', '_')}_weighted_inverted_index"
        collection = db[collection_name]
        collection.delete_many({})
        collection.insert_many([
            {"term": term, "docs": doc_list}
            for term, doc_list in weighted_index.items()
        ])

        safe_name = dataset_path.replace("/", "__")
        os.makedirs(os.path.join("db", safe_name), exist_ok=True)
        joblib.dump(weighted_index, os.path.join("db", safe_name, "weighted_inverted_index.joblib"))

        logger.info(f"✅ Weighted inverted index saved to MongoDB for dataset '{dataset_path}'.")
        return {"status": "Weighted inverted index created successfully", "terms_count": len(weighted_index)}

    except FileNotFoundError as e:
        return {"error": str(e)}
