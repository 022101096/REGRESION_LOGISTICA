import pandas as pd
import numpy as np
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from src.preprocessing import prepare_data, save_processed_data, create_tfidf_vectorizer

print("=== REENTRENAMIENTO BILINGÜE (ES + EN) ===\n")

# Cargar dataset combinado bilingüe
df = pd.read_csv('data/raw/spam_bilingual_combined.csv')
print(f"Dataset cargado: {len(df)} muestras")
print(f"Distribución: {df['label'].value_counts().to_dict()}")

# Preparar datos (usa el nuevo preprocessing con SnowballStemmer + stopwords ES+EN)
X_train, X_test, y_train, y_test, vectorizer, scaler = prepare_data(
    df, 
    text_col='text', 
    label_col='label',
    test_size=0.2,
    random_state=42,
    max_features=5000  # Aumentado para vocabulario bilingüe
)

# Entrenar Regresión Logística
print("\nEntrenando Regresión Logística...")
model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='liblinear',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'f1': float(f1_score(y_test, y_pred)),
    'roc_auc': float(roc_auc_score(y_test, y_prob))
}

print("\n=== MÉTRICAS EN TEST ===")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Guardar modelo y artefactos
import os
os.makedirs('models', exist_ok=True)

joblib.dump(model, 'models/logistic_regression.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Guardar metadata
metadata = {
    'model_type': 'LogisticRegression',
    'language': 'bilingual_es_en',
    'stemmer': 'SnowballStemmer(spanish)',
    'stopwords': 'english + spanish',
    'max_features': 5000,
    'ngram_range': [1, 2],
    'train_samples': int(X_train.shape[0]),
    'test_samples': int(X_test.shape[0]),
    'metrics': metrics
}
with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# También guardar datos procesados
save_processed_data(X_train, X_test, y_train, y_test, vectorizer, scaler)

print("\n✅ Modelo y artefactos guardados en models/")
print("✅ Datos procesados guardados en data/processed/")