# import os
# import joblib
# from sklearn.feature_extraction.text import TfidfVectorizer
# from fastapi import APIRouter
# from pydantic import BaseModel

# from backend.database.connection import get_mongo_connection
# from backend.services.text_processing_service import processed_text,spacy_tokenizer
# from backend.logger_config import logger

# from tqdm import tqdm
# import time

# for i in tqdm(range(10)):
#     time.sleep(0.5)

# router = APIRouter()

# class TFIDFRequest(BaseModel):
#     dataset_path: str

# @router.post("/tfidf/build")
# def create_tfidf_vector_from_corpus(request: TFIDFRequest):
#     dataset_path = request.dataset_path
#     logger.info('Start creating TF-IDF vectors')

#     db = get_mongo_connection()
#     collection_name = dataset_path.replace("/", "_")
#     collection = db[collection_name]

#     all_doc_ids = []
#     all_processed_texts = []

#     cursor = collection.find({}, {"_id": 0, "doc_id": 1, "text": 1})
#     for doc in cursor:
#         if "doc_id" in doc and "text" in doc:
#             text = doc["text"]
#             if isinstance(text, str) and text.strip():
#                 all_doc_ids.append(doc["doc_id"])
#                 all_processed_texts.append(processed_text(doc["text"]))

#     logger.info(f"Vectorizing {len(all_doc_ids)} documents")
#     vectorizer = TfidfVectorizer(
#         lowercase=False,
#         preprocessor=None,
#         tokenizer=str.split,
#         token_pattern=None,
#         max_df=0.8,
#         min_df=5,
#         ngram_range=(1, 2),
#         sublinear_tf=True,
#         max_features=30000
#     )


#     tfidf_matrix = vectorizer.fit_transform(all_processed_texts)

#     safe_name = dataset_path.replace("/", "__")
#     db_dir = os.path.join("db", safe_name)
#     os.makedirs(db_dir, exist_ok=True)

#     joblib.dump(tfidf_matrix, os.path.join(db_dir, "tfidf_matrix.joblib"), compress=3)
#     joblib.dump(vectorizer.vocabulary_, os.path.join(db_dir, "vocabulary.joblib"))
#     joblib.dump(vectorizer, os.path.join(db_dir, "vectorizer.joblib"))
#     joblib.dump(all_doc_ids, os.path.join(db_dir, "doc_ids.joblib"))
#     joblib.dump(all_processed_texts, os.path.join(db_dir, "all_texts.joblib"))

#     logger.info(f"TF-IDF created and saved for {dataset_path}")

#     total_docs = collection.count_documents({})
#     accepted_docs = len(all_doc_ids)
#     skipped_docs = total_docs - accepted_docs

#     logger.info(f"📊 Total: {total_docs}, Accepted: {accepted_docs}, Skipped: {skipped_docs}")

#     return {"status": "TF-IDF vectorizer built successfully", "documents_processed": len(all_doc_ids)}

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

from fastapi import APIRouter
from pydantic import BaseModel
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from backend.services.text_processing_service import processed_text, TextProcessor
from backend.database.connection import get_mongo_connection
from backend.logger_config import logger
import pandas as pd

router = APIRouter()
processor = TextProcessor()
def preprocess_for_tfidf(text):
        return processed_text(text, processor)

class BuildTFIDFRequest(BaseModel):
    dataset_path: str   # اسم مجموعة MongoDB مثل: lotte/lifestyle/dev

@router.post("/tfidf/build")
def build_tfidf(request: BuildTFIDFRequest):
    logger.info('Start creating TF-IDF vectors')
    dataset_path = request.dataset_path
    # 👇 تحميل البيانات من Mongo بدل pd.read_csv
    db = get_mongo_connection()
    collection_name = dataset_path.replace("/", "_")
    collection = db[collection_name]

    texts = []
    pids = []

    cursor = collection.find({}, {"_id": 0, "doc_id": 1, "text": 1})
    for doc in cursor:
        if "doc_id" in doc and "text" in doc and isinstance(doc["text"], str):
            pids.append(doc["doc_id"])
            texts.append(doc["text"])

    # 👇 حطّينا البيانات في DataFrame لتبقى بقية الكود كما هي
    data = pd.DataFrame({"pid": pids, "text": texts})
    data.dropna(subset=['text'], inplace=True)
    
    tfidf_vectorizer = TfidfVectorizer(
        preprocessor=preprocess_for_tfidf,
        max_df=0.5,
        min_df=1
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'])

    safe_name = dataset_path.replace("/", "__")
    db_dir = os.path.join("db", safe_name)
    os.makedirs(db_dir, exist_ok=True)
    joblib.dump(tfidf_vectorizer, os.path.join(db_dir, "vectorizer.joblib"))
    joblib.dump(tfidf_matrix, os.path.join(db_dir, "tfidf_matrix.joblib"))
    joblib.dump(data, os.path.join(db_dir, "docs.joblib"))
    
    logger.info(f"TF-IDF created and saved for {dataset_path}")
    return {"status": f"✅ TF-IDF built successfully on dataset: {dataset_path}", "documents": len(data)}