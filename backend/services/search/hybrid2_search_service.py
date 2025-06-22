import os
import sys
import joblib
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from backend.services.text_processing_service import processed_text
from backend.logger_config import logger

bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid2_search(query: str, dataset_path: str, top_k=50, candidate_k=500, w_tfidf=0.4, w_bm25=0.6):
    logger.info("Running hybrid search with reranking: %s", query)
    safe_name = dataset_path.replace("/", "__")
    db_dir = os.path.join("db", safe_name)

    tfidf_matrix = joblib.load(os.path.join(db_dir, "tfidf_matrix.joblib"))
    vectorizer = joblib.load(os.path.join(db_dir, "vectorizer.joblib"))
    bm25_model: BM25Okapi = joblib.load(os.path.join(db_dir, "bm25_model.joblib"))
    bert_embeddings = joblib.load(os.path.join(db_dir, "bert_embeddings.joblib"))
    doc_ids = joblib.load(os.path.join(db_dir, "doc_ids.joblib"))
    texts = joblib.load(os.path.join(db_dir, "all_texts.joblib"))  # النصوص المعالجة

    query_processed = processed_text(query)
    query_vec = vectorizer.transform([query_processed])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    bm25_scores = bm25_model.get_scores(query_processed.split())

    # ترشيح مرشحين (inverted index filter)
    top_tfidf_idx = np.argsort(tfidf_scores)[::-1][:candidate_k]
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:candidate_k]
    candidate_indices = list(set(top_tfidf_idx).union(set(top_bm25_idx)))

    logger.info(f"Candidate pool size: {len(candidate_indices)}")

    # BERT Bi-Encoder score لحساب تشابه سريع
    query_embedding = bi_encoder.encode(query_processed, normalize_embeddings=True)
    bert_scores = np.dot(bert_embeddings[candidate_indices], query_embedding)

    # دمج السكورات المبدئية
    combined_scores = (
        w_tfidf * tfidf_scores[candidate_indices] +
        w_bm25 * bm25_scores[candidate_indices] +
        0.0 * bert_scores  # BERT ما بيدخل هون، بس ممكن ترفعي وزنه لو حبيتي
    )

    # ترتيب مبدئي للمرشحين قبل CrossEncoder
    rerank_pool = sorted(
        zip(candidate_indices, combined_scores),
        key=lambda x: x[1],
        reverse=True
    )[:candidate_k]
    print(f"Type of texts: {type(texts)}")
    print(f"Type of texts[0]: {type(texts[0])}")
    print(f"Example texts[0]: {texts[0]}")

    # إعداد أزواج (query, doc) لـ CrossEncoder
    cross_inputs = [(query, texts[i]) for i, _ in rerank_pool]
    ce_scores = cross_encoder.predict(cross_inputs)

    # ترتيب نهائي
    final_results = sorted(
        zip([doc_ids[i] for i, _ in rerank_pool], ce_scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return final_results

if __name__ == "__main__":
    query = "attention mechanism in transformers"
    results = hybrid2_search(query, "vaswani", top_k=10)
    for rank, (doc_id, score) in enumerate(results, 1):
        print(f"{rank}. {doc_id} (score: {score:.4f})")
