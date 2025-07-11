# backend/services/suggestions/models_cache.py

from sentence_transformers import SentenceTransformer
from happytransformer import HappyTextToText, TTSettings

bert_model = SentenceTransformer("all-MiniLM-L6-v2")
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
tt_settings = TTSettings(do_sample=False)
