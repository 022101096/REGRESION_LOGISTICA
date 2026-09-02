import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Descargar stopwords si no existen
for lang in ['english', 'spanish']:
    try:
        stopwords.words(lang)
    except LookupError:
        nltk.download(lang, quiet=True)

STOP_WORDS_EN = set(stopwords.words('english'))
STOP_WORDS_ES = set(stopwords.words('spanish'))
STOP_WORDS = STOP_WORDS_EN | STOP_WORDS_ES
STEMMER = SnowballStemmer('spanish')


def clean_text(text: str) -> str:
    """
    Limpieza básica de texto para SMS.
    """
    if not isinstance(text, str):
        return ""
    
    # Minúsculas
    text = text.lower()
    
    # Remover URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remover números de teléfono y códigos
    text = re.sub(r'\b\d{5,}\b', '', text)
    
    # Remover puntuación y caracteres especiales (mantener espacios)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remover números aislados
    text = re.sub(r'\b\d+\b', '', text)
    
    # Tokenizar, remover stopwords, stemming
    tokens = text.split()
    tokens = [STEMMER.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    
    return ' '.join(tokens)


def create_tfidf_vectorizer(max_features: int = 3000, 
                            ngram_range: tuple = (1, 2),
                            min_df: int = 2,
                            max_df: float = 0.95) -> TfidfVectorizer:
    """
    Crea y configura el vectorizador TF-IDF.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        strip_accents='unicode'
    )


def prepare_data(df: pd.DataFrame, 
                 text_col: str = 'v2', 
                 label_col: str = 'v1',
                 test_size: float = 0.2,
                 random_state: int = 42,
                 max_features: int = 3000):
    """
    Pipeline completo de preparación de datos.
    
    Returns:
        X_train, X_test, y_train, y_test, vectorizer, scaler
    """
    # Limpiar textos
    print("Limpiando textos...")
    df['clean_text'] = df[text_col].apply(clean_text)
    
    # Filtrar vacíos
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    
    # Labels: spam=1, ham=0
    y = (df[label_col] == 'spam').astype(int)
    X_text = df['clean_text']
    
    # Split estratificado
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # TF-IDF
    print("Vectorizando con TF-IDF...")
    vectorizer = create_tfidf_vectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    # Escalado (opcional para LR, pero ayuda a convergencia)
    scaler = StandardScaler(with_mean=False)  # with_mean=False para sparse
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Distribución train - Spam: {y_train.mean():.2%}, Ham: {(1-y_train.mean()):.2%}")
    print(f"Distribución test  - Spam: {y_test.mean():.2%}, Ham: {(1-y_test.mean()):.2%}")
    
    return X_train, X_test, y_train, y_test, vectorizer, scaler


def save_processed_data(X_train, X_test, y_train, y_test, 
                        vectorizer, scaler, 
                        output_dir: str = 'data/processed'):
    """
    Guarda todos los artefactos de preprocesamiento.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(X_train, f'{output_dir}/X_train.pkl')
    joblib.dump(X_test, f'{output_dir}/X_test.pkl')
    joblib.dump(y_train, f'{output_dir}/y_train.pkl')
    joblib.dump(y_test, f'{output_dir}/y_test.pkl')
    joblib.dump(vectorizer, f'{output_dir}/tfidf_vectorizer.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    print(f"Artefactos guardados en {output_dir}/")


def load_processed_data(input_dir: str = 'data/processed'):
    """
    Carga artefactos de preprocesamiento.
    """
    X_train = joblib.load(f'{input_dir}/X_train.pkl')
    X_test = joblib.load(f'{input_dir}/X_test.pkl')
    y_train = joblib.load(f'{input_dir}/y_train.pkl')
    y_test = joblib.load(f'{input_dir}/y_test.pkl')
    vectorizer = joblib.load(f'{input_dir}/tfidf_vectorizer.pkl')
    scaler = joblib.load(f'{input_dir}/scaler.pkl')
    return X_train, X_test, y_train, y_test, vectorizer, scaler


def preprocess_for_inference(text: str, vectorizer, scaler):
    """
    Preprocesa un texto individual para inferencia.
    """
    clean = clean_text(text)
    X = vectorizer.transform([clean])
    X = scaler.transform(X)
    return X, clean