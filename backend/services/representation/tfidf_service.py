import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import APIRouter
from pydantic import BaseModel

from backend.database.connection import get_mongo_connection
from backend.services.text_processing_service import processed_text
from backend.logger_config import logger

router = APIRouter()

class TFIDFRequest(BaseModel):
    dataset_path: str

@router.post("/tfidf/build")
def create_tfidf_vector_from_corpus(request: TFIDFRequest):
    dataset_path = request.dataset_path
    logger.info('Start creating TF-IDF vectors')

    db = get_mongo_connection()
    collection_name = dataset_path.replace("/", "_")
    collection = db[collection_name]

    all_doc_ids = []
    all_processed_texts = []

    cursor = collection.find({}, {"_id": 0, "doc_id": 1, "text": 1})
    for doc in cursor:
        if "doc_id" in doc and "text" in doc:
            all_doc_ids.append(doc["doc_id"])
            all_processed_texts.append(processed_text(doc["text"]))

    logger.info(f"Vectorizing {len(all_processed_texts)} documents")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_processed_texts)

    safe_name = dataset_path.replace("/", "__")
    db_dir = os.path.join("db", safe_name)
    os.makedirs(db_dir, exist_ok=True)

    joblib.dump(tfidf_matrix, os.path.join(db_dir, "tfidf_matrix.joblib"), compress=3)
    joblib.dump(vectorizer.vocabulary_, os.path.join(db_dir, "vocabulary.joblib"))
    joblib.dump(vectorizer, os.path.join(db_dir, "vectorizer.joblib"))
    joblib.dump(all_doc_ids, os.path.join(db_dir, "doc_ids.joblib"))
    joblib.dump(all_processed_texts, os.path.join(db_dir, "all_texts.joblib"))

    logger.info(f"TF-IDF created and saved for {dataset_path}")
    return {"status": "TF-IDF vectorizer built successfully", "documents_processed": len(all_doc_ids)}
