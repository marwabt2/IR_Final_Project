import os
import joblib
import numpy as np
import faiss
from fastapi import APIRouter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from backend.database.connection import get_mongo_connection
from backend.services.text_processing_service import processed_text,TextProcessor
from backend.logger_config import logger

router = APIRouter()
processor = TextProcessor()
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")  # optimized for retrieval


class BERTFaissRequest(BaseModel):
    dataset_path: str

def embed_text(text):
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding

def build_faiss_index(embeddings, index_path):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity via inner product
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)

@router.post("/bert_faiss/build")
def create_bert_faiss(request: BERTFaissRequest):
    dataset_path = request.dataset_path
    logger.info("Start BERT FAISS generation for dataset: %s", dataset_path)

    db = get_mongo_connection()
    collection_name = dataset_path.replace("/", "_")
    collection = db[collection_name]

    safe_name = dataset_path.replace("/", "__")
    db_dir = os.path.join("db", safe_name)
    os.makedirs(db_dir, exist_ok=True)

    all_embeddings = []
    all_doc_ids = []

    documents = list(collection.find({}, {"_id": 0, "doc_id": 1, "text": 1}))
    for doc in documents:
        if "text" in doc and "doc_id" in doc:
            text_processed = processed_text(doc["text"],processor)
            embedding = embed_text(text_processed)
            all_embeddings.append(embedding)
            all_doc_ids.append(doc["doc_id"])

    if not all_embeddings:
        return {"error": "No documents found in dataset."}

    all_embeddings_np = np.array(all_embeddings)

    joblib.dump(all_embeddings_np, os.path.join(db_dir, "bert_embeddings.joblib"))
    joblib.dump(all_doc_ids, os.path.join(db_dir, "bert_doc_ids.joblib"))

    faiss_index_path = os.path.join(db_dir, "bert_faiss.index")
    build_faiss_index(all_embeddings_np, faiss_index_path)

    logger.info("FAISS index saved to: %s", faiss_index_path)
    logger.info("BERT embeddings saved at: %s", db_dir)
    logger.info("Finished all embeddings for dataset: %s", dataset_path)

    return {
        "status": "BERT FAISS index created successfully",
        "documents_processed": len(all_doc_ids),
        "faiss_index_path": faiss_index_path
    }

class BERTEmbeddingRequest(BaseModel):
    dataset_path: str

@router.post("/bert_embeddings/build")
def create_bert_embeddings(request: BERTEmbeddingRequest):
    dataset_path = request.dataset_path
    logger.info("Start BERT embedding generation for dataset: %s", dataset_path)

    db = get_mongo_connection()
    collection_name = dataset_path.replace("/", "_")
    collection = db[collection_name]

    safe_name = dataset_path.replace("/", "__")
    db_dir = os.path.join("db", safe_name)
    os.makedirs(db_dir, exist_ok=True)

    all_embeddings = []
    all_doc_ids = []

    documents = list(collection.find({}, {"_id": 0, "doc_id": 1, "text": 1}))
    for doc in documents:
        if "text" in doc and "doc_id" in doc:
            text_processed = processed_text(doc["text"], processor)
            embedding = embed_text(text_processed)
            all_embeddings.append(embedding)
            all_doc_ids.append(doc["doc_id"])

    if not all_embeddings:
        return {"error": "No documents found in dataset."}

    all_embeddings_np = np.array(all_embeddings)

    joblib.dump(all_embeddings_np, os.path.join(db_dir, "bert_embeddings.joblib"))
    joblib.dump(all_doc_ids, os.path.join(db_dir, "bert_doc_ids.joblib"))

    logger.info("BERT embeddings saved at: %s", db_dir)
    logger.info("Finished all embeddings for dataset: %s", dataset_path)

    return {
        "status": "BERT embeddings created successfully",
        "documents_processed": len(all_doc_ids),
        "embeddings_path": os.path.join(db_dir, "bert_embeddings.joblib")
    }