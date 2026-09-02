import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import json
from datetime import datetime


def train_logistic_regression(X_train, y_train, 
                              C: float = 1.0,
                              penalty: str = 'l2',
                              solver: str = 'lbfgs',
                              max_iter: int = 2000,
                              class_weight: str = 'balanced',
                              random_state: int = 42):
    """
    Entrena modelo de Regresión Logística.
    """
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    
    print("Entrenando Regresión Logística...")
    model.fit(X_train, y_train)
    
    # Verificar convergencia
    if not model.n_iter_[0] < max_iter:
        print(f"WARNING: Modelo no convergio en {max_iter} iteraciones")
    else:
        print(f"OK: Modelo convergio en {model.n_iter_[0]} iteraciones")
    
    return model


def get_feature_importance(model, vectorizer, top_n: int = 20):
    """
    Obtiene las features más importantes (coeficientes) para cada clase.
    """
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    
    # Top spam (coeficientes positivos)
    spam_idx = np.argsort(coef)[-top_n:][::-1]
    # Top ham (coeficientes negativos)
    ham_idx = np.argsort(coef)[:top_n]
    
    spam_features = pd.DataFrame({
        'feature': feature_names[spam_idx],
        'coefficient': coef[spam_idx],
        'class': 'spam'
    })
    
    ham_features = pd.DataFrame({
        'feature': feature_names[ham_idx],
        'coefficient': coef[ham_idx],
        'class': 'ham'
    })
    
    return pd.concat([spam_features, ham_features]).reset_index(drop=True)


def cross_validate_model(model, X, y, cv: int = 5):
    """
    Validación cruzada estratificada.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='f1', n_jobs=-1)
    print(f"CV F1-Score: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    return scores


def save_model(model, vectorizer, scaler, metrics: dict, 
               output_dir: str = 'models'):
    """
    Guarda modelo, vectorizador, scaler y metadata.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(model, f'{output_dir}/logistic_regression.pkl')
    joblib.dump(vectorizer, f'{output_dir}/tfidf_vectorizer.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    
    metadata = {
        'model_type': 'LogisticRegression',
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': {
            'C': model.C,
            'penalty': model.penalty,
            'solver': model.solver,
            'max_iter': model.max_iter,
            'class_weight': model.class_weight
        },
        'metrics': metrics,
        'n_features': model.coef_.shape[1],
        'n_iter': int(model.n_iter_[0])
    }
    
    with open(f'{output_dir}/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Modelo y metadata guardados en {output_dir}/")


def load_model(model_dir: str = 'models'):
    """
    Carga modelo y artefactos.
    """
    model = joblib.load(f'{model_dir}/logistic_regression.pkl')
    vectorizer = joblib.load(f'{model_dir}/tfidf_vectorizer.pkl')
    scaler = joblib.load(f'{model_dir}/scaler.pkl')
    
    with open(f'{model_dir}/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return model, vectorizer, scaler, metadata


def predict_with_explanation(model, vectorizer, scaler, text: str, top_n: int = 5):
    """
    Predice y devuelve explicabilidad (top features que influyeron).
    """
    from src.preprocessing import clean_text
    
    clean = clean_text(text)
    X = vectorizer.transform([clean])
    X = scaler.transform(X)
    
    prob = model.predict_proba(X)[0, 1]
    pred = model.predict(X)[0]
    
    # Explicabilidad: contribución = tfidf_value * coef
    feature_indices = X.nonzero()[1]
    if len(feature_indices) > 0:
        features = vectorizer.get_feature_names_out()[feature_indices]
        tfidf_vals = X.toarray()[0][feature_indices]
        coefs = model.coef_[0][feature_indices]
        contributions = tfidf_vals * coefs
        
        # Top spam / ham
        sorted_idx = np.argsort(contributions)
        top_spam_idx = sorted_idx[-top_n:][::-1]
        top_ham_idx = sorted_idx[:top_n]
        
        explanation = {
            'spam': [{'feature': features[i], 'contribution': float(contributions[i])} 
                     for i in top_spam_idx],
            'ham': [{'feature': features[i], 'contribution': float(contributions[i])} 
                    for i in top_ham_idx]
        }
    else:
        explanation = {'spam': [], 'ham': []}
    
    return {
        'prediction': 'spam' if pred == 1 else 'ham',
        'probability_spam': float(prob),
        'clean_text': clean,
        'explanation': explanation
    }