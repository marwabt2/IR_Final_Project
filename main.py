import sys
import os

from backend.services.search import hybrid_search_service
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

from turtle import pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
from pydantic import BaseModel
from backend.services import hybrid2_search_service,hybrid3_search_service

app = FastAPI()
dataset_path='antique/train'

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class QueryInput(BaseModel):
    text: str
    dataset_path: str


@app.post("/process-text")
async def process_text(data: QueryInput):
    
    results = hybrid3_search_service.hybrid3_search_with_rerank(data.text, data.dataset_path)
    formatted = [
        {
            "doc_id": doc.get("doc_id"),
            "score": round(doc.get("similarity_score", 0), 4),
            "text": doc.get("text")[:300] + "..." if doc.get("text") else ""
        }
        for doc in results
    ]
    return JSONResponse(content={"processed_text": formatted})
    