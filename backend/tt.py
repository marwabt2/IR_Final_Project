import csv
import logging
import nltk
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import inflect
import re
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import unicodedata
import contractions
from textblob import TextBlob
import spacy

# Load spaCy model for advanced text processing
# nlp = spacy.load('en_core_web_sm')
# with open(r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\common_words.txt", 'r',
#           encoding='utf-8') as file:
#     words_to_remove = file.read().splitlines()


class TextProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.inflect_engine = inflect.engine()
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    
    def clean_text(self, text, words_to_remove):
        words = text.split()
        cleaned_words = [word for word in words if word not in words_to_remove]
        cleaned_text = ' '.join(cleaned_words)
        return cleaned_text

    def number_to_words(self, text):
        words = self.tokenizer.tokenize(text)
        converted_words = []
        for word in words:
            # التحقق مما إذا كان النص يمثل رقمًا
            if word.replace('.', '', 1).isdigit():  # إزالة النقطة العشرية قبل فحص الرقم
                converted_words.append(word)
            else:
                if word.isdigit():
                    try:
                        num = int(word)
                        if num <= 999999999999999:  # تحقق من طول الرقم
                            converted_word = self.inflect_engine.number_to_words(word)
                            converted_words.append(converted_word)
                        else:
                            converted_words.append("[Number Out of Range]")
                    except inflect.NumOutOfRangeError:
                        converted_words.append("[Number Out of Range]")
                else:
                    converted_words.append(word)
        return ' '.join(converted_words)

    def remove_html_tags(self, text):
        try:
            # Check if the input text contains HTML tags before parsing
            if '<' in text and '>' in text:
                return BeautifulSoup(text, "html.parser").get_text()
            else:
                # If no HTML tags are found, return the original text
                return text
        except MarkupResemblesLocatorWarning:
            # Handle the warning gracefully
            logging.warning("MarkupResemblesLocatorWarning: The input looks more like a filename than markup.")
            # Return the original text if unable to parse as HTML
            return text

    def normalize_unicode(self, text):
        return unicodedata.normalize("NFKD", text)

    def expand_contractions(self, text):
        return contractions.fix(text)

    def cleaned_text(self, text):
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def normalization_example(self, text):
        return text.lower()

    def stemming_example(self, text):
        words = self.tokenizer.tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)

    def lemmatization_example(self, text):
        words = self.tokenizer.tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    def remove_stopwords(self, text):
        words = self.tokenizer.tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)

    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)

    def remove_urls(self, text):
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    def remove_special_characters_and_emojis(self, text):
        return re.sub(r'[^A-Za-z0-9\s]+', '', text)


    def replace_synonyms(self, text):
        words = self.tokenizer.tokenize(text)
        synonym_words = [self.get_synonym(word) for word in words]
        return ' '.join(synonym_words)

    def get_synonym(self, word):
        synonyms = nltk.corpus.wordnet.synsets(word)
        if synonyms:
            return synonyms[0].lemmas()[0].name()
        return word

    def handle_negations(self, text):
        words = self.tokenizer.tokenize(text)
        negated_text = []
        negate = False
        for word in words:
            if word.lower() in ['not', "n't"]:
                negate = True
            elif negate:
                negated_text.append(f"NOT_{word}")
                negate = False
            else:
                negated_text.append(word)
        return ' '.join(negated_text)

    def remove_non_english_words(self, text):
        words = self.tokenizer.tokenize(text)
        english_words = [word for word in words if wordnet.synsets(word)]
        return ' '.join(english_words)


def process_text(text, processor):
    if text is None:
        return text
    text = processor.cleaned_text(text)
    text = processor.normalization_example(text)
    text = processor.stemming_example(text)
    text = processor.lemmatization_example(text)
    text = processor.remove_stopwords(text)
    text = processor.number_to_words(text)
    text = processor.remove_punctuation(text)
    # text = processor.clean_text(text, words_to_remove)
    text = processor.expand_contractions(text)
    text = processor.normalize_unicode(text)
    text = processor.handle_negations(text)
    text = processor.remove_urls(text)
    return text


import pandas as pd 
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, average_precision_score
import numpy as np
import sys

from textblob import TextBlob


sys.path.append('.')


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

def save_dataset(docs, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for pid, text in enumerate(docs, start=1):
            file.write(f"{pid}\t{text}\n")


def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path, delimiter='\t', header=None, names=['pid', 'text'])
    except pd.errors.ParserError as e:
        print(f"Error reading the dataset file: {e}")
        sys.exit(1)
    return data


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


def process_texts(texts, processor):
    processed_texts = []
    for text in texts:
        if isinstance(text, str):
            processed_text = process_text(text, processor)
            processed_texts.append(processed_text)
        else:
            print("Skipping non-string value:", text)
    return processed_texts


def vectorize_texts(texts, processor):
    vectorizer = TfidfVectorizer(preprocessor=lambda x: process_text(x, processor), max_df=0.5, min_df=1)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError as e:
        print(f"Error during TF-IDF vectorization: {e}")
        print(f"Sample texts: {texts[:5]}")
        sys.exit(1)

    return tfidf_matrix, vectorizer


def get_documents_for_query(query, tfidf_matrix, processor, vectorizer, data):
    processed_query = process_text(query, processor)
    query_vector = vectorizer.transform([processed_query])
    cosine_similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()
    n = 10
    top_documents_indices = cosine_similarities.argsort()[-n:][::-1]
    top_documents = data.iloc[top_documents_indices]
    return top_documents, cosine_similarities[top_documents_indices]


if __name__ == '__main__':
    processor = TextProcessor()

    dataset_path = r'C:\Users\USER\Desktop\lifestyle\dev\collection.tsv'
    data = load_dataset(dataset_path)
    data.dropna(subset=['text'], inplace=True)

    if 'text' not in data.columns:
        print("exit")
        sys.exit(1)

    tfidf_matrix, vectorizer = vectorize_texts(data['text'], processor)
    queries_paths = [r'C:\Users\USER\Desktop\lifestyle\dev\qas.search.jsonl']
    queries = load_queries(queries_paths)

    all_precisions = []
    all_recalls = []
    all_map_scores = []
    all_mrrs = []

    for query in queries:
        if 'query' in query:

            top_documents, cosine_similarities = get_documents_for_query(query['query'], tfidf_matrix, processor,
                                                                         vectorizer, data)

            relevance = np.zeros(len(data))
            found_match = False

            for pid in query.get('answer_pids', []):
                indices = np.where(data['pid'].astype(str) == str(pid))[0]
                if len(indices) > 0:
                    relevance[indices] = 1
                    found_match = True

            if not found_match:
                print(f"⚠️ لا يوجد أي pid مطابق في data['pid'] للاستعلام: {query['query']}")
                continue

            relevantOrNot = relevance[top_documents.index]
            retrievedDocument = cosine_similarities

            if relevantOrNot.sum() == 0:
                print(f"⚠️ لم يتم استرجاع أي وثيقة ذات صلة للاستعلام: {query['query']}")
                continue
            precision, recall = calculate_precision_recall(relevantOrNot, retrievedDocument)
            all_precisions.append(precision)
            all_recalls.append(recall)

            map_score = calculate_map_score(relevantOrNot, retrievedDocument)
            all_map_scores.append(map_score)

            mrr = calculate_mrr(relevantOrNot)
            all_mrrs.append(mrr)

    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_map_score = np.mean(all_map_scores)
    avg_mrr = np.mean(all_mrrs)

    print(
        f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average MAP Score: {avg_map_score}, Average MRR: {avg_mrr}")
