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

app = FastAPI()
app.include_router(tfidf_search_router)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

