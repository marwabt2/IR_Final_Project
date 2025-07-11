import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from collections import defaultdict
import ir_datasets
import httpx
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, average_precision_score
from backend.database.connection import get_mongo_connection
from fastapi.responses import JSONResponse

from backend.services.text_processing_service import router as text_processing_router
from backend.services.representation.tfidf_service import router as tfidf_router
from backend.services.representation.bert_service import router as bert_router
from backend.services.representation.bm25_service import router as bm25_router
from backend.services.representation.hybrid_service import router as hybrid_router
from backend.services.representation.weighted_inverted_index_service import router as weighted_router
from backend.services.representation.bm25_weighted_inverted_index_service import router as bm25_weighted_router

from backend.services.search.tfidf_search_service import router as tfidf_search_router
from backend.services.search.bert_search_service import router as bert_search_router
from backend.services.search.bm25_search_service import router as bm25_search_router
from backend.services.search.hybrid_search_service import router as hybrid_search_router

from backend.services.clustering_service import router as clustering_router
app = FastAPI()

# تسجيل الروترات
app.include_router(text_processing_router)
app.include_router(tfidf_router)
app.include_router(bert_router)
app.include_router(bm25_router)
app.include_router(hybrid_router)
app.include_router(weighted_router)
app.include_router(bm25_weighted_router)

app.include_router(tfidf_search_router)
app.include_router(bert_search_router)
app.include_router(bm25_search_router)
app.include_router(hybrid_search_router)

app.include_router(clustering_router)

class DatasetPathRequest(BaseModel):
    dataset_path: str

all_precisions = []
all_recalls = []
all_map_scores = []
all_mrrs = []

def calculate_precision_recall(relevantOrNot, retrievedDocument, threshold=0.5):
    binaryResult = (retrievedDocument >= threshold).astype(int)
    precision = precision_score(relevantOrNot, binaryResult, average='micro')
    recall = recall_score(relevantOrNot, binaryResult, average='micro')
    return precision, recall


def calculate_map_score(relevantOrNot, retrievedDocument):
    return average_precision_score(relevantOrNot, retrievedDocument, average='micro')

def calculate_mrr(y_true):
    rank_position = np.where(y_true == 1)[0]
    if len(rank_position) == 0:
        return 0
    else:
        return 1 / (rank_position[0] + 1)  # +1 because rank positions are 1-based

def load_queries(queries_paths):
    queries = []
    for file_path in queries_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    query = json.loads(line.strip())
                    if 'query' in query:
                        queries.append(query)
                except json.JSONDecodeError:
                    print(f"Skipping invalid line in {file_path}: {line}")
    return queries

async def evaluate_search(request: DatasetPathRequest, search_api_url: str):
    import time
    start_time = time.time()  # ⏱️ بداية حساب الزمن
    safe_name = request.dataset_path.replace("/", "__")
    db = get_mongo_connection()
    collection_name = request.dataset_path.replace("/", "_")
    collection = db[collection_name]

    texts = []
    pids = []

    cursor = collection.find({}, {"_id": 0, "doc_id": 1, "text": 1})
    for doc in cursor:
        if "doc_id" in doc and "text" in doc and isinstance(doc["text"], str):
            pids.append(str(doc["doc_id"]))  # ← cast to string directly
            texts.append(doc["text"])

    data = pd.DataFrame({"pid": pids, "text": texts})
    data.dropna(subset=['text'], inplace=True)
    data["pid"] = data["pid"].astype(str)  # ← تأكيد
    queries_paths = ''
    if request.dataset_path == 'lotte/lifestyle/dev/forum':
        queries_paths = r'C:\Users\USER\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\qas.search.jsonl'
    if request.dataset_path == 'antique/train':
        queries_paths = r'C:\Users\USER\.ir_datasets\antique\test\Answers.jsonl'

    queries = load_queries([queries_paths])

    async with httpx.AsyncClient(timeout=300.0) as client:
        for query in queries:
            if 'query' in query:
                response = await client.post(
                    search_api_url,
                    json={
                        "dataset_path": request.dataset_path,
                        "query": query["query"],
                        "top_n": 10
                    }
                )

                if response.status_code != 200:
                    print(f"❌ Search API failed: {response.status_code}")
                    continue

                response_json = response.json()
                top_documents = pd.DataFrame(response_json["top_documents"])
                cosine_similarities = np.array(response_json["cosine_similarities"])
                retrieved_indices = response_json["top_documents_indices"]

                relevance = np.zeros(len(data))

                for pid in query.get('answer_pids', []):
                    pid_str = str(pid)
                    indices = np.where(data['pid'] == pid_str)[0]
                    relevance[indices] = 1

                retrievedDocument = cosine_similarities
                relevantOrNot = relevance[retrieved_indices]

                if relevantOrNot.sum() == 0:
                    print(f"Query skipped – no relevant documents found for: {query['query']}")
                    continue

                precision, recall = calculate_precision_recall(relevantOrNot, retrievedDocument)
                all_precisions.append(precision)
                all_recalls.append(recall)

                map_score = calculate_map_score(relevantOrNot, retrievedDocument)
                all_map_scores.append(map_score)

                mrr = calculate_mrr(relevantOrNot)
                all_mrrs.append(mrr)

    if len(all_precisions) == 0:
        print("⚠️ No valid queries evaluated. Check PIDs matching and dataset content.")
        return

    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_map_score = np.mean(all_map_scores)
    avg_mrr = np.mean(all_mrrs)

        # ⏱️ حساب الزمن
    elapsed_time = time.time() - start_time

    print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average MAP Score: {avg_map_score}, Average MRR: {avg_mrr}")
    return JSONResponse(content={
    "execution_time_seconds": round(elapsed_time, 3),
    "average_precision": avg_precision,
    "average_recall": avg_recall,
    "average_map_score": avg_map_score,
    "average_mrr": avg_mrr
})
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

@app.post("/tfidf/eval/clustering")
async def search_with_clustering(request: DatasetPathRequest):
    return await evaluate_search(request, "http://localhost:8000/tfidf/search/clustering")

@app.post("/normal/bert/eval")
async def normal_bert_eval(request: DatasetPathRequest):
    return await evaluate_search(request, "http://localhost:8000/normal/bert/search")







