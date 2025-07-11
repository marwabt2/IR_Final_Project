# backend/services/suggestions/suggestions_engine.py
import time 
import os
import re
import numpy as np
import nltk
import joblib
import torch
import requests
import re
import urllib.parse
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from symspellpy import SymSpell, Verbosity
from nltk.corpus import wordnet as wn
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.services.suggestions.models_cache import bert_model, happy_tt, tt_settings
from backend.services.text_processing_service import processed_text, TextProcessor

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
rerank_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")

def fetch_duckduckgo_suggestions(query):
    try:
        url = "https://duckduckgo.com/ac/"
        params = {"q": query, "type": "list"}
        response = requests.get(url, params=params)
        data = response.json()
        if isinstance(data, list):
            return [item["phrase"] for item in data if isinstance(item, dict) and "phrase" in item]
        else:
            return []
    except Exception as e:
        print(f"❌ Error fetching DuckDuckGo suggestions for: {query} → {e}")
        return []


def fetch_google_suggestions(query):
        try:
            query = re.sub(r"[^\w\s]", "", query)  
            url = f"https://suggestqueries.google.com/complete/search?client=firefox&q={urllib.parse.quote(query)}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and len(data) > 1:
                return data[1][:10]  
            else:
                return []
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching Google suggestions for: {query} → {e}")
            return []
        
class SuggestionEngine:
    def __init__(self, dataset: str): 
        db_map = {
            "lotte/lifestyle/dev/forum": "queries_lifestyle",
            "antique/train": "queries_antique",
        }

        collection_name = db_map.get(dataset)
        if not collection_name:
            raise ValueError("Unsupported dataset")

        self.collection_name = collection_name
        self.cache_dir = os.path.join(os.path.dirname(__file__), "cache", collection_name)
        os.makedirs(self.cache_dir, exist_ok=True)

        client = MongoClient("mongodb://localhost:27017/")
        db = client["query_db"]
        self.collection = db[collection_name]
        self.user_query_collection = db[collection_name]
        self._load_user_query_log()

        self._load_cached_data()

        self.sym_spell = SymSpell(max_dictionary_edit_distance=2)
        dict_path = os.path.join(os.path.dirname(__file__), "frequency_dictionary_en_82_765.txt")
        self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)


    def _load_user_query_log(self):
        print(f"[Autocomplete] Loading user queries from collection: {self.user_query_collection.name}")
        self.user_queries = [
            doc["query"].strip() for doc in self.user_query_collection.find({}, {"_id": 0, "query": 1})
            if "query" in doc and isinstance(doc["query"], str) and doc["query"].strip()
        ]
        self.user_queries = sorted(list(set(self.user_queries)), key=str.lower)
    def _load_cached_data(self):
        queries_path = os.path.join(self.cache_dir, "queries.joblib")
        tfidf_path = os.path.join(self.cache_dir, "tfidf.joblib")
        bert_path = os.path.join(self.cache_dir, "bert.joblib")  

        if all(os.path.exists(p) for p in [queries_path, tfidf_path, bert_path]):
            self.queries = joblib.load(queries_path)
            self.vectorizer, self.tfidf_matrix = joblib.load(tfidf_path)
            self.bert_embeddings = joblib.load(bert_path)
        else:
            print(f"[Cache] Generating embeddings for: {self.collection_name}")
            
            # تحميل الكويريات من MongoDB
            docs = list(self.collection.find({}, {"_id": 0, "query": 1}))
            self.queries = [doc["query"] for doc in docs]
            joblib.dump(self.queries, queries_path)

            # إنشاء وتمثيل TF-IDF
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = self.vectorizer.fit_transform(self.queries)
            joblib.dump((self.vectorizer, self.tfidf_matrix), tfidf_path)

            # تمثيلات BERT
            self.bert_embeddings = bert_model.encode(
                self.queries, convert_to_tensor=False, show_progress_bar=False
            )
            joblib.dump(self.bert_embeddings, bert_path)


    # def _load_cached_data(self):
    #     queries_path = os.path.join(self.cache_dir, "queries.pkl")
    #     tfidf_path = os.path.join(self.cache_dir, "tfidf.pkl")
    #     bert_path = os.path.join(self.cache_dir, "bert.npy")

    #     if all(os.path.exists(p) for p in [queries_path, tfidf_path, bert_path]):
    #         self.queries = joblib.load(queries_path)
    #         self.vectorizer, self.tfidf_matrix = joblib.load(tfidf_path)
    #         self.bert_embeddings = np.load(bert_path)
    #     else:
    #         print(f"[Cache] Generating embeddings for: {self.collection_name}")
    #         docs = list(self.collection.find({}, {"_id": 0, "query": 1}))
    #         self.queries = [doc["query"] for doc in docs]
    #         joblib.dump(self.queries, queries_path)

    #         self.vectorizer = TfidfVectorizer(stop_words="english")
    #         self.tfidf_matrix = self.vectorizer.fit_transform(self.queries)
    #         joblib.dump((self.vectorizer, self.tfidf_matrix), tfidf_path)

    #         self.bert_embeddings = bert_model.encode(self.queries, convert_to_tensor=False, show_progress_bar=False)
    #         np.save(bert_path, self.bert_embeddings)

    # def correct_spelling(self, query: str) -> str:
    #     corrected = []
    #     for word in query.split():
    #         suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
    #         corrected.append(suggestions[0].term if suggestions else word)
    #     return " ".join(corrected)

    def correct_spelling(self, query: str) -> str:
        if len(query.strip().split()) > 1:
            suggestions = self.sym_spell.lookup_compound(query, max_edit_distance=2)
            return suggestions[0].term if suggestions else query
        else:
            corrected = []
            for word in query.split():
                if len(word) <= 2:
                    corrected.append(word)
                    continue
                suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                corrected.append(suggestions[0].term if suggestions else word)
            return " ".join(corrected)

    def correct_grammar(self, query: str) -> str:
        result = happy_tt.generate_text(f"grammar: {query}", args=tt_settings)
        return result.text.strip()

    def expand_query(self, query: str) -> str:
        words = nltk.word_tokenize(query)
        expanded = set(words)

        for word in words:
            for syn in wn.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace("_", " ")
                    if synonym != word:
                        expanded.add(synonym)

        return " ".join(expanded)
    

    def rerank(self, query: str, candidates: List[str]) -> List[str]:
        scores = []
        for cand in candidates:
            tokens = rerank_tokenizer(query, cand, return_tensors="pt", truncation=True)
            with torch.no_grad():
                score = rerank_model(**tokens).logits[0].item()
            scores.append(score)
        sorted_items = sorted(zip(scores, candidates), reverse=True)
        return [text for _, text in sorted_items]

 
    # def suggest_queries(self, user_query: str, top_k=5) -> Tuple[List[Tuple[str, float]], str, str]:
    #     # 1. تصحيح الإملاء والنحو
    #     corrected = self.correct_spelling(user_query)
    #     grammar_corrected = self.correct_grammar(corrected)

    #     # 2. التوسيع والمعالجة
    #     expanded = self.expand_query(grammar_corrected)
    #     processor = TextProcessor()
    #     processed_query = processed_text(expanded, processor)

    #     # 3. استخراج تمثيلات BERT و TF-IDF
    #     query_embedding = bert_model.encode(processed_query, convert_to_tensor=False, show_progress_bar=False)
    #     bert_scores = cosine_similarity([query_embedding], self.bert_embeddings)[0]

    #     tfidf_query = self.vectorizer.transform([processed_query])
    #     tfidf_scores = cosine_similarity(tfidf_query, self.tfidf_matrix)[0]

    #     # 4. الدمج: BERT له وزن أعلى
    #     final_scores = 0.7 * bert_scores + 0.3 * tfidf_scores

    #     # 5. استخراج أفضل المرشحين (نأخذ أكثر من top_k لنعطي rerank مجالًا)
    #     top_indices = np.argsort(final_scores)[::-1][:top_k * 15]
    #     candidates = [self.queries[i] for i in top_indices if final_scores[i] > 0.15]

    #     # 6. إعادة الترتيب (rerank) بالاعتماد على النموذج الخاص بك
    #     reranked = self.rerank(grammar_corrected, candidates)

    #     # 7. أخذ أفضل top_k مباشرة من reranked
    #     top_suggestions = [(r, 1.0) for r in reranked[:top_k]]

    #     return top_suggestions, corrected, grammar_corrected
    
    def suggest_queries(self, user_query: str, top_k=5) -> Tuple[List[Tuple[str, float]], str, str]:
        # 1. تصحيح الإملاء والنحو
        corrected = self.correct_spelling(user_query)
        grammar_corrected = self.correct_grammar(corrected)

        # 2. جلب الاقتراحات من Google و DuckDuckGo
        google_suggestions = fetch_google_suggestions(grammar_corrected)
        duckduckgo_suggestions = fetch_duckduckgo_suggestions(grammar_corrected)

        # 3. دمج الاقتراحات مع الاقتراحات الداخلية
        all_suggestions = google_suggestions + duckduckgo_suggestions
        all_suggestions = list(set(all_suggestions))  # إزالة التكرار

        # 4. التوسيع والمعالجة
        expanded = self.expand_query(grammar_corrected)
        processor = TextProcessor()
        processed_query = processed_text(expanded, processor)

        # 5. استخراج تمثيلات BERT و TF-IDF
        query_embedding = bert_model.encode(processed_query, convert_to_tensor=False, show_progress_bar=False)
        bert_scores = cosine_similarity([query_embedding], self.bert_embeddings)[0]

        tfidf_query = self.vectorizer.transform([processed_query])
        tfidf_scores = cosine_similarity(tfidf_query, self.tfidf_matrix)[0]

        # 6. الدمج: BERT له وزن أعلى
        final_scores = 0.7 * bert_scores + 0.3 * tfidf_scores

        # 7. استخراج أفضل المرشحين
        top_indices = np.argsort(final_scores)[::-1][:top_k * 15]
        candidates = [self.queries[i] for i in top_indices if final_scores[i] > 0.15]

        # 8. دمج الاقتراحات الخارجية مع الاقتراحات الداخلية
        candidates += all_suggestions  # دمج اقتراحات Google و DuckDuckGo

        # 9. إعادة الترتيب (rerank) بالاعتماد على النموذج الخاص بك
        reranked = self.rerank(grammar_corrected, candidates)

        # 10. أخذ أفضل top_k مباشرة من reranked
        top_suggestions = [(r, 1.0) for r in reranked[:top_k]]

        return top_suggestions, corrected, grammar_corrected

    

    def autocomplete(self, prefix: str, limit=5) -> List[str]:
        start_time = time.time()  
        prefix = prefix.strip().lower()
        if len(prefix) < 2:
            return []

        results = []
        count = 0

        for query in self.user_queries:
            q = query.lower()
            if q.startswith(prefix):
                results.append(query)
                count += 1
            if count >= limit:
                break

        elapsed = (time.time() - start_time) * 1000 
        print(f"[autocomplete] Time taken: {elapsed:.2f} ms")  

        return results
