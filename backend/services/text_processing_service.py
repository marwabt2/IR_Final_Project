import logging
import nltk
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import inflect
import re
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import unicodedata
import contractions
import spacy

# Optional: load spaCy for advanced NLP
# nlp = spacy.load('en_core_web_sm')

# Load domain-specific stopwords (common_words.txt)
# with open(r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\common_words.txt", 'r', encoding='utf-8') as file:
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
        return ' '.join(cleaned_words)

    def number_to_words(self, text):
        words = self.tokenizer.tokenize(text)
        converted_words = []
        for word in words:
            if word.replace('.', '', 1).isdigit():
                converted_words.append(word)
            else:
                if word.isdigit():
                    try:
                        num = int(word)
                        if num <= 999999999999999:
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
            if '<' in text and '>' in text:
                return BeautifulSoup(text, "html.parser").get_text()
            return text
        except MarkupResemblesLocatorWarning:
            logging.warning("MarkupResemblesLocatorWarning: input looks more like a filename than markup.")
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
        return ' '.join([self.stemmer.stem(word) for word in words])

    def lemmatization_example(self, text):
        words = self.tokenizer.tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])

    def remove_stopwords(self, text):
        words = self.tokenizer.tokenize(text)
        return ' '.join([word for word in words if word.lower() not in self.stop_words])

    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)

    def remove_urls(self, text):
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    def remove_special_characters_and_emojis(self, text):
        return re.sub(r'[^A-Za-z0-9\s]+', '', text)

    def replace_synonyms(self, text):
        words = self.tokenizer.tokenize(text)
        return ' '.join([self.get_synonym(word) for word in words])

    def get_synonym(self, word):
        synonyms = nltk.corpus.wordnet.synsets(word)
        if synonyms:
            return synonyms[0].lemmas()[0].name()
        return word

    def handle_negations(self, text):
        words = self.tokenizer.tokenize(text)
        result = []
        negate = False
        for word in words:
            if word.lower() in ['not', "n't"]:
                negate = True
            elif negate:
                result.append(f"NOT_{word}")
                negate = False
            else:
                result.append(word)
        return ' '.join(result)

    def remove_non_english_words(self, text):
        words = self.tokenizer.tokenize(text)
        return ' '.join([word for word in words if wordnet.synsets(word)])


# Global instance (reused)
processor = TextProcessor()


def processed_text(text):
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
