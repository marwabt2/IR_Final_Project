from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from collections import defaultdict
import ir_datasets
import httpx

from backend.services.representation.tfidf_service import router as tfidf_router
from backend.services.representation.bert_service import router as bert_router
from backend.services.representation.hybrid_service import router as hybrid_router
from backend.services.representation.weighted_inverted_index_service import router as weighted_router

from backend.services.search.tfidf_search_service import router as tfidf_search_router
from backend.services.search.bert_search_service import router as bert_search_router
from backend.services.search.hybrid_search_service import router as hybrid_search_router

app = FastAPI()

# تسجيل الروترات
app.include_router(tfidf_router)
app.include_router(bert_router)
app.include_router(hybrid_router)
app.include_router(weighted_router)

app.include_router(tfidf_search_router)
app.include_router(bert_search_router)
app.include_router(hybrid_search_router)

class DatasetPathRequest(BaseModel):
    dataset_path: str


def load_qrels(dataset):
    qrels = defaultdict(set)
    for qrel in dataset.qrels_iter():
        if qrel.relevance >= 1:
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

async def evaluate_search(request: DatasetPathRequest, search_api_url: str):
    dataset = ir_datasets.load(request.dataset_path)
    queries = list(dataset.queries_iter())
    qrels = load_qrels(dataset)

    results = {"MAP": 0.0, "MRR": 0.0, "P@10": 0.0, "Recall": 0.0, "evaluated": 0}

    async with httpx.AsyncClient() as client:
        for query in queries:
            relevant = qrels.get(query.query_id)
            if not relevant:
                continue
            payload = {"dataset_path": request.dataset_path, "query": query.text}
            try:
                response = await client.post(search_api_url, json=payload)
                response.raise_for_status()
                data = response.json()
                retrieved = [doc["doc_id"] for doc in data.get("results", [])]
            except Exception as e:
                print(f"⚠️ Query {query.query_id} failed: {e}")
                continue

            results["MAP"] += average_precision(retrieved, relevant)
            results["MRR"] += reciprocal_rank(retrieved, relevant)
            results["P@10"] += precision_at_k(retrieved, relevant, 10)
            results["Recall"] += recall(retrieved, relevant)
            results["evaluated"] += 1

    if results["evaluated"] == 0:
        raise HTTPException(status_code=500, detail="❌ No queries evaluated.")

    for metric in ["MAP", "MRR", "P@10", "Recall"]:
        results[metric] /= results["evaluated"]

    return {"status": "success", "metrics": results}

@app.post("/tfidf/eval")
async def tfidf_eval(request: DatasetPathRequest):
    return await evaluate_search(request, "http://localhost:8000/tfidf/search")

@app.post("/bert/eval")
async def bert_eval(request: DatasetPathRequest):
    return await evaluate_search(request, "http://localhost:8000/bert/search")

@app.post("/bm25/eval")
async def bm25_eval(request: DatasetPathRequest):
    return await evaluate_search(request, "http://localhost:8000/bm25/search")

@app.post("/hybrid/eval")
async def hybrid_eval(request: DatasetPathRequest):
    return await evaluate_search(request, "http://localhost:8000/hybrid/search")

@app.post("/hybrid2/eval")
async def hybrid2_eval(request: DatasetPathRequest):
    return await evaluate_search(request, "http://localhost:8000/hybrid2/search")
