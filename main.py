import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

from turtle import pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
from pydantic import BaseModel
import httpx
from fastapi.responses import JSONResponse
from backend.services.search.tfidf_search_service import router as tfidf_search_router
from backend.services.search.bert_search_service import router as bert_search_router
from backend.services.search.bm25_search_service import router as bm25_search_router
from backend.services.search.hybrid_search_service import router as hybrid_search_router
from backend.services.search.bert_search_query_ref_service  import router as bert_search_query_ref_router

from backend.services.clustering_service import router as clustering_router
from backend.services.suggestions.suggestions_router import router as suggestions_router

app = FastAPI()
app.include_router(tfidf_search_router)
app.include_router(bert_search_router)
app.include_router(hybrid_search_router)
app.include_router(bm25_search_router)
app.include_router(clustering_router)
app.include_router(suggestions_router)
app.include_router(bert_search_query_ref_router)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/advanced", response_class=HTMLResponse)
async def advanced_features_page(request: Request):
    return templates.TemplateResponse("advanced.html", {"request": request})
@app.get("/cluster", response_class=HTMLResponse)
async def clustering_page(request: Request):
    return templates.TemplateResponse("cluster.html", {"request": request})

@app.get("/vector_store", response_class=HTMLResponse)
async def vector_store_page(request: Request):
    return templates.TemplateResponse("vector_store.html", {"request": request})