# CONTEXTO DEL PROYECTO - Clasificador Spam Bilingüe (ES/EN)

## 📋 Resumen
Proyecto de clasificación binaria SMS Spam vs No Spam usando Regresión Logística + TF-IDF, desplegado en Streamlit Cloud.

## 🏗️ Estructura clave
```
Aplicacion-regresion/
├── app/
│   ├── streamlit_app.py       # App principal (bilingüe ES/EN)
│   └── requirements.txt       # Dependencias para deploy
├── models/                    # Artefactos del modelo (en GitHub)
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
├── src/
│   └── preprocessing.py       # Preprocesamiento bilingüe (SnowballStemmer ES + stopwords ES/EN)
├── data/raw/                  # Datasets combinados
│   ├── spam_bilingual_combined.csv   # 25,731 muestras (ES + EN)
│   ├── spam_en_combined.csv
│   └── spam_es_combined.csv
└── retrain_bilingual.py       # Script de reentrenamiento
```

## ⚙️ Configuración técnica

### Preprocesamiento (identico en training + inference)
- **Stemmer**: `SnowballStemmer('spanish')` (funciona para ES + EN)
- **Stopwords**: Unión de `stopwords.words('english')` + `stopwords.words('spanish')`
- **TF-IDF**: max_features=5000, ngram_range=(1,2), sublinear_tf=True
- **Scaler**: StandardScaler(with_mean=False)
- **Modelo**: LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear')

### Datasets usados
| Dataset | Muestras | Fuente |
|---------|----------|--------|
| SMS Spam Collection (EN) | 5,572 | UCI/Kaggle |
| Multilingüe traducido (ES) | 5,572 | HuggingFace dbarbedillo |
| Sintético ES (tanaos) | 15,016 | HuggingFace tanaos |
| **Total combinado** | **25,731** | - |

## 🚀 Despliegue en Streamlit Cloud

### URL pública
`https://<usuario>-regresion-logistica.streamlit.app` (se genera automáticamente)

### Pasos deploy
1. Push a `main` en GitHub → auto-redeploy
2. O manual: share.streamlit.io → New app → repo → branch `main` → file `app/streamlit_app.py`

### Requisitos en repo
- `models/*.pkl` NO en .gitignore (necesarios para inference)
- `requirements.txt` en raíz y en `app/`
- Python 3.10+ en Advanced settings

## 🔧 Comandos útiles

### Reentrenar modelo
```bash
cd D:\2026UAC\IA\Aplicacion-regresion
venv\Scripts\python retrain_bilingual.py
git add models/ && git commit -m "Retrain" && git push
```

### Ejecutar local
```bash
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
streamlit run app/streamlit_app.py
```

### Test rápido modelo
```bash
venv\Scripts\python -c "
import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

model = joblib.load('models/logistic_regression.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
scaler = joblib.load('models/scaler.pkl')

STOP_WORDS = set(stopwords.words('english')) | set(stopwords.words('spanish'))
STEMMER = SnowballStemmer('spanish')

def clean(t):
    t = t.lower()
    t = re.sub(r'http\S+|www\S+|https\S+', '', t)
    t = re.sub(r'\b\d{5,}\b', '', t)
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\b\d+\b', '', t)
    return ' '.join([STEMMER.stem(w) for w in t.split() if w not in STOP_WORDS and len(w)>1])

for msg in ['recibe un iphone gratis', 'FREE iPhone giveaway!', 'el profesor nos quiere aprobar']:
    c = clean(msg)
    X = scaler.transform(vectorizer.transform([c]))
    p = model.predict_proba(X)[0,1]
    print(f'{msg[:35]:35s} -> {\"SPAM\" if p>0.5 else \"HAM\"} ({p:.3f})')
"
```

## 📊 Métricas actuales (Test set)
| Métrica | Valor |
|---------|-------|
| Accuracy | 0.908 |
| Precision (Spam) | 0.862 |
| Recall (Spam) | 0.894 |
| F1-Score (Spam) | 0.877 |
| ROC-AUC | 0.957 |

## ⚠️ Limitaciones conocidas
1. **"el profesor nos quiere aprobar iphone gratis"** → HAM (falso negativo: términos académicos no vistos como spam)
2. Spam inglés clásico ("FREE iPhone giveaway") a veces clasifica como HAM por dominio diferente
3. Necesita más datos de spam ES con contexto educativo/académico

## 🔄 Para mejorar
- Agregar más muestras spam ES: "profesor aprueba", "nota gratis", "examen facil"
- Balancear mejor EN/ES (ahora 70% ES sintético)
- Probar modelos más robustos (XGBoost, BERT multilingüe)

## 📝 Última actualización
- Modelo reentrenado bilingüe: SnowballStemmer + stopwords ES/EN
- 25,731 muestras combinadas
- Push a GitHub: commit `c819451`
- Streamlit Cloud auto-redeploy activado