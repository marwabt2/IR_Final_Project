# backend/services/suggestions/suggestions_router.py

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from backend.services.suggestions.suggestions_engine import SuggestionEngine

router = APIRouter()


lifestyle_engine = SuggestionEngine("lotte/lifestyle/dev/forum")
antique_engine = SuggestionEngine("antique/train")

@router.get("/api/suggest")
async def suggest_api(q: str = "", dataset: str = ""):
    if not q or not dataset:
        return JSONResponse(status_code=400, content={"error": "Missing query or dataset"})

    if dataset == "lotte/lifestyle/dev/forum":
        engine = lifestyle_engine
    elif dataset == "antique/train":
        engine = antique_engine
    else:
        return JSONResponse(status_code=400, content={"error": "Unknown dataset"})

    suggestions, corrected, grammar_corrected = engine.suggest_queries(q)
    return {
        "suggestions": suggestions,
        "corrected": corrected,
        "grammar_corrected": grammar_corrected,
    }

@router.get("/api/autocomplete")
async def autocomplete_api(prefix: str = "", dataset: str = ""):
    if not prefix or not dataset:
        return JSONResponse(status_code=400, content={"error": "Missing prefix or dataset"})

    if dataset == "lotte/lifestyle/dev/forum":
        engine = lifestyle_engine
    elif dataset == "antique/train":
        engine = antique_engine
    else:
        return JSONResponse(status_code=400, content={"error": "Unknown dataset"})

    return {"results": engine.autocomplete(prefix)}
