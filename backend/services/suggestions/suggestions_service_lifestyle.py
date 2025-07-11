### backend/services/suggestions/suggestions_service_lifestyle.py

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import Optional
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from symspellpy import SymSpell, Verbosity
from sentence_transformers import SentenceTransformer
from happytransformer import HappyTextToText, TTSettings
import numpy as np
import nltk
import os
import re

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

router = APIRouter()

# الاتصال بقاعدة البيانات
client = MongoClient("mongodb://localhost:27017/")
db = client["query_db"]
collection = db["queries_lifestyle"]
docs = list(collection.find({}, {"_id": 0, "query": 1}))
queries = [doc["query"] for doc in docs]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(queries)

# BERT model
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
bert_embeddings = bert_model.encode(queries, convert_to_tensor=True)

# Grammar correction
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
tt_settings = TTSettings(do_sample=False)

# Spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2)
dict_path = os.path.join(os.path.dirname(__file__), "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)

def correct_spelling(query):
    corrected = []
    for word in query.split():
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected.append(suggestions[0].term)
        else:
            corrected.append(word)
    return " ".join(corrected)

def correct_grammar(query):
    result = happy_tt.generate_text(f"grammar: {query}", args=tt_settings)
    return result.text.strip()

def expand_query(query):
    from nltk.corpus import wordnet as wn
    words = nltk.word_tokenize(query)
    expanded = set(words)
    for word in words:
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))
    return " ".join(expanded)

def suggest_queries(user_query, top_k=5):
    corrected_query = correct_spelling(user_query)
    grammar_corrected = correct_grammar(corrected_query)
    expanded = expand_query(grammar_corrected)

    query_embedding = bert_model.encode(expanded, convert_to_tensor=True)
    bert_scores = cosine_similarity([query_embedding], bert_embeddings)[0]

    tfidf_query = vectorizer.transform([grammar_corrected])
    tfidf_scores = cosine_similarity(tfidf_query, tfidf_matrix)[0]

    final_scores = 0.5 * bert_scores + 0.5 * tfidf_scores
    top_indices = np.argsort(final_scores)[::-1][:top_k]
    suggestions = [(queries[i], float(final_scores[i])) for i in top_indices if final_scores[i] > 0.2]

    return suggestions, corrected_query, grammar_corrected

def autocomplete(prefix, limit=5):
    regex = f"^{re.escape(prefix)}"
    matches = collection.aggregate([
        {"$match": {"query": {"$regex": regex, "$options": "i"}}},
        {"$group": {"_id": "$query", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit}
    ])
    return [doc["_id"] for doc in matches]

@router.get("/api/suggest")
async def suggest_api(q: Optional[str] = ""):
    if not q:
        return JSONResponse(status_code=400, content={"error": "missing query"})
    suggestions, corrected, grammar_corrected = suggest_queries(q)
    return {
        "suggestions": suggestions,
        "corrected": corrected,
        "grammar_corrected": grammar_corrected
    }

@router.get("/api/autocomplete")
async def autocomplete_api(prefix: Optional[str] = ""):
    if not prefix:
        return JSONResponse(status_code=400, content={"error": "missing prefix"})
    return {"results": autocomplete(prefix)}
