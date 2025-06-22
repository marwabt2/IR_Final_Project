import string
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from spellchecker import SpellChecker
from backend.logger_config import logger

def processed_text(text: str) -> str:
    text = remove_urls(text)
    text = _remove_punctuations(text)
    tokens = _get_words_tokenize(text)
    tokens = _lowercase_tokens(tokens)
    tokens = _remove_stopwords(tokens)
    tokens = _filter_foreign_characters(tokens)
    tokens = preprocess_and_lemmatize(tokens)
    return ' '.join(tokens)

def processed_query_fun(text: str) -> str:
    text = remove_urls(text)
    text = _remove_punctuations(text)
    tokens = _get_words_tokenize(text)
    tokens = _lowercase_tokens(tokens)
    tokens = _remove_stopwords(tokens)
    tokens = _filter_foreign_characters(tokens)
    tokens = spell_check(tokens)
    tokens = preprocess_and_lemmatize(tokens)
    return ' '.join(tokens)

def _get_words_tokenize(text: str) -> list:
    return word_tokenize(text)

def _remove_stopwords(tokens: list) -> list:
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def _remove_punctuations(text: str) -> str:
    pattern = r"(?<!\d)\.(?!\d)|[^\w\s.]"
    return re.sub(pattern, " ", text)

def remove_urls(text: str) -> str:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def _lowercase_tokens(tokens: list) -> list:
    return [token.lower() for token in tokens]

def _filter_foreign_characters(tokens: list) -> list:
    return [token for token in tokens if all(c in string.printable for c in token)]

def get_wordnet_pos(word: str) -> str:
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_and_lemmatize(tokens: list) -> list:
    lemmatizer = WordNetLemmatizer()
    pos_tagged = pos_tag(tokens)
    return [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token, tag in pos_tagged]

def _stem_tokens(tokens: list) -> list:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def spell_check(tokens: list) -> list:
    spell = SpellChecker(language='en')
    misspelled = spell.unknown(tokens)
    corrections = {}
    for word in misspelled:
        candidate = spell.correction(word)
        corrections[word] = candidate if candidate else word
    return [corrections.get(word, word) for word in tokens if word]
