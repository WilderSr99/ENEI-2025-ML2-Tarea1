"""
preprocessing.py — Funciones de limpieza y preprocesamiento de texto.
Compatible con los datasets del Assignment 1 (Disaster Tweets).
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos requeridos solo si faltan
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def preprocess_text(text: str) -> str:
    """
    Limpieza robusta de tweets:
    - Convierte a minúsculas
    - Elimina URLs, menciones @
    - Quita el símbolo '#' pero mantiene la palabra (ej: '#fire' -> 'fire')
    - Elimina signos de puntuación
    - Tokeniza, lematiza y elimina stopwords en inglés

    Retorna: string limpio
    """
    if not isinstance(text, str):
        return ""

    # Normalización básica
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)   # URLs
    text = re.sub(r"@\w+", " ", text)                      # menciones
    text = text.replace("#", " ")                          # elimina '#' pero conserva palabra
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenización y lematización
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words and tok.strip()]

    # Unión final y limpieza de espacios múltiples
    cleaned = re.sub(r"\s+", " ", " ".join(tokens)).strip()
    return cleaned
