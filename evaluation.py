# evaluate_hybrid.py
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

import ir_datasets
from collections import defaultdict
from backend.services.search.hybrid_search_service import hybrid_search  
from backend.services.search.tfidf_search_service import tfidf_search
from backend.services.search.bert_search_service import bert_search
def load_qrels(dataset):
    qrels = defaultdict(set)
    for qrel in dataset.qrels_iter():
        if qrel.relevance >= 1:  # ✅ فقط الوثائق ذات الصلاحية المقبولة أو الممتازة
            qrels[qrel.query_id].add(qrel.doc_id)
    return qrels

def average_precision(retrieved, relevant):
    if not relevant:
        return 0.0
    score = 0.0
    num_hits = 0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / len(relevant)

def reciprocal_rank(retrieved, relevant):
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1 / (i + 1)
    return 0.0

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = [doc_id for doc_id in retrieved_k if doc_id in relevant]
    return len(hits) / k

def recall(retrieved, relevant):
    if not relevant:
        return 0.0
    hits = [doc_id for doc_id in retrieved if doc_id in relevant]
    return len(hits) / len(relevant)

def evaluate(dataset_path="vaswani", top_k=10):
    dataset = ir_datasets.load(dataset_path)
    queries = list(dataset.queries_iter())
    qrels_dict = load_qrels(dataset)

    total_ap = 0.0
    total_rr = 0.0
    total_prec10 = 0.0
    total_recall = 0.0
    evaluated = 0

    for query in queries:
        query_id = query.query_id
        text = query.text
        relevant = qrels_dict.get(query_id)

        if not relevant:
            continue

        try:
            results = hybrid_search(text, dataset_path)
            # results = tfidf_search(text, dataset_path)
            # results = bert_faiss_search(text, dataset_path)

            retrieved = [doc["doc_id"] for doc in results]

        except Exception as e:
            print(f"⚠️ خطأ بالاستعلام {query_id}: {e}")
            continue

        total_ap += average_precision(retrieved, relevant)
        total_rr += reciprocal_rank(retrieved, relevant)
        total_prec10 += precision_at_k(retrieved, relevant, 10)
        total_recall += recall(retrieved, relevant)
        evaluated += 1

    if evaluated == 0:
        print("❌ لم يتم تقييم أي استعلام.")
        return

    print(f"✅ تم تقييم {evaluated} استعلام")
    print(f"🎯 MAP: {total_ap / evaluated:.4f}")
    print(f"📌 MRR: {total_rr / evaluated:.4f}")
    print(f"🔟 Precision@10: {total_prec10 / evaluated:.4f}")
    print(f"📈 Recall: {total_recall / evaluated:.4f}")

if __name__ == "__main__":
    evaluate()
