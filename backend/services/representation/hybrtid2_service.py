import os
import sys
import joblib
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

from backend.logger_config import logger

from tqdm import tqdm
import time

for i in tqdm(range(10)):
    time.sleep(0.5)

def load_component(dataset_path: str, name: str):
    safe_name = dataset_path.replace("/", "__")
    path = os.path.join("db", safe_name, name)
    return joblib.load(path)

def create_hybrid_representation2(dataset_path: str):
    logger.info("Loading components for hybrid representation: %s", dataset_path)
    safe_name = dataset_path.replace("/", "__")
    db_dir = os.path.join("db", safe_name)

    tfidf_matrix = load_component(dataset_path, "tfidf_matrix.joblib")
    vectorizer = load_component(dataset_path, "vectorizer.joblib")
    bm25_model = load_component(dataset_path, "bm25_model.joblib")
    bert_embeddings = load_component(dataset_path, "bert_embeddings.joblib")
    doc_ids = load_component(dataset_path, "doc_ids.joblib")

    joblib.dump(tfidf_matrix, os.path.join(db_dir, "hybrid_tfidf_matrix.joblib"))
    joblib.dump(vectorizer, os.path.join(db_dir, "hybrid_vectorizer.joblib"))
    joblib.dump(bm25_model, os.path.join(db_dir, "hybrid_bm25_model.joblib"))
    joblib.dump(bert_embeddings, os.path.join(db_dir, "hybrid_bert_embeddings.joblib"))
    joblib.dump(doc_ids, os.path.join(db_dir, "hybrid_doc_ids.joblib"))

    logger.info("Hybrid representation saved for dataset: %s", dataset_path)

if __name__ == "__main__":
    create_hybrid_representation2("vaswani")
