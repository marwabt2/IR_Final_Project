# backend/services/query_enhancement_service.py

import nltk
import json
import os
from symspellpy import SymSpell, Verbosity
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk import pos_tag, word_tokenize

# تحميل الموارد اللازمة لمرة واحدة فقط
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# تحميل نموذج التصحيح الإملائي
sym_spell = SymSpell(max_dictionary_edit_distance=2)

current_dir = os.path.dirname(os.path.abspath(__file__))
dict_path = os.path.join(current_dir, "frequency_dictionary_en_82_765.txt")

if not os.path.exists(dict_path):
    raise FileNotFoundError(f"❌ ملف القاموس غير موجود: {dict_path}")

sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)

# تحميل نموذج تصحيح القواعد
grammar_tokenizer = AutoTokenizer.from_pretrained("./models/t5_grammar")
grammar_model = AutoModelForSeq2SeqLM.from_pretrained("./models/t5_grammar")

from backend.services.text_processing_service import TextProcessor
processor = TextProcessor()

class QueryEnhancer:
    def __init__(self):
        self.sym_spell = sym_spell
        self.tokenizer = grammar_tokenizer
        self.model = grammar_model

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

    def correct_grammar_conservatively(self, query: str) -> str:
        if len(query.split()) <= 5:
            return query
        input_text = f"grammar: {query}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True)
        output_ids = self.model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
        corrected = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return corrected if corrected.strip().lower() != query.strip().lower() else query

    def expand_query_minimally(self, query: str, max_total_expansions=3) -> str:
        words = word_tokenize(query)
        pos_tags = pos_tag(words)
        expanded = list(words)
        added_set = set(w.lower() for w in words)
        total_expanded = 0

        def get_wordnet_pos(tag):
            if tag.startswith("J"):
                return wn.ADJ
            elif tag.startswith("V"):
                return wn.VERB
            elif tag.startswith("N"):
                return wn.NOUN
            elif tag.startswith("R"):
                return wn.ADV
            return None

        for word, tag in pos_tags:
            word_lower = word.lower()
            if word_lower in processor.stop_words or len(word_lower) < 3:
                continue

            wn_pos = get_wordnet_pos(tag)
            if not wn_pos:
                continue

            synsets = wn.synsets(word_lower, pos=wn_pos)
            if not synsets:
                continue

            for lemma in synsets[0].lemmas():
                synonym = lemma.name().replace("_", " ").lower()

                if (
                    synonym == word_lower or
                    synonym in added_set or
                    word_lower in synonym or
                    synonym in query.lower() or
                    synonym in processor.stop_words or
                    len(synonym) < 3 or
                    synonym.isnumeric()
                ):
                    continue

                expanded.append(synonym)
                added_set.add(synonym)
                total_expanded += 1
                break

            if total_expanded >= max_total_expansions:
                break

        return " ".join(expanded)

# إنشاء نسخة جاهزة للاستدعاء
query_enhancer = QueryEnhancer()
