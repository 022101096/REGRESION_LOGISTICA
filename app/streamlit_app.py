import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Descargar stopwords si no existen (necesario en Streamlit Cloud)
for lang in ['english', 'spanish']:
    try:
        nltk.data.find(f'corpora/stopwords/{lang}')
    except LookupError:
        nltk.download(lang, quiet=True)

# Configuración de página
st.set_page_config(
    page_title="Clasificador Spam vs No Spam",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #2E86AB 0%, #E94F37 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .spam-alert {
        background: #fff3f3;
        border-left-color: #E94F37 !important;
    }
    .ham-alert {
        background: #f0fff4;
        border-left-color: #2E86AB !important;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #2E86AB 0%, #E94F37 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        font-size: 1.1rem;
    }
    .stButton > button:hover {
        opacity: 0.9;
    }
    .feature-tag {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.2rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-family: monospace;
    }
    .feature-spam { background: #ffeaea; color: #c0392b; border: 1px solid #f5b7b1; }
    .feature-ham { background: #eafaf1; color: #1e8e3e; border: 1px solid #a9dfbf; }
</style>
""", unsafe_allow_html=True)

# ===========================
# FUNCIONES DE CARGA Y PROCESAMIENTO
# ===========================

@st.cache_resource
def load_artifacts():
    """Carga modelo, vectorizador y scaler (cacheado)."""
    try:
        model = joblib.load("models/logistic_regression.pkl")
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return model, vectorizer, scaler
    except FileNotFoundError as e:
        st.error(f"❌ No se encontraron los archivos del modelo: {e}")
        st.info("Ejecuta primero los notebooks para entrenar y guardar el modelo.")
        st.stop()

# Preprocesamiento IDÉNTICO al entrenamiento
STOP_WORDS_EN = set(stopwords.words('english'))
STOP_WORDS_ES = set(stopwords.words('spanish'))
STOP_WORDS = STOP_WORDS_EN | STOP_WORDS_ES
STEMMER = SnowballStemmer('spanish')

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\b\d{5,}\b', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    tokens = text.split()
    tokens = [STEMMER.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)

def get_explanation(vectorizer, model, scaler, text: str, top_n: int = 5):
    """Obtiene top features que influyeron en la predicción."""
    clean = clean_text(text)
    X = vectorizer.transform([clean])
    X = scaler.transform(X)
    
    feature_indices = X.nonzero()[1]
    if len(feature_indices) == 0:
        return {'spam': [], 'ham': []}
    
    features = vectorizer.get_feature_names_out()[feature_indices]
    tfidf_vals = X.toarray()[0][feature_indices]
    coefs = model.coef_[0][feature_indices]
    contributions = tfidf_vals * coefs
    
    sorted_idx = np.argsort(contributions)
    top_spam_idx = sorted_idx[-top_n:][::-1]
    top_ham_idx = sorted_idx[:top_n]
    
    return {
        'spam': [{'feature': features[i], 'contribution': float(contributions[i])} 
                 for i in top_spam_idx if contributions[i] > 0],
        'ham': [{'feature': features[i], 'contribution': float(contributions[i])} 
                for i in top_ham_idx if contributions[i] < 0]
    }

# ===========================
# CARGAR ARTEFACTOS
# ===========================
model, vectorizer, scaler = load_artifacts()

# ===========================
# INTERFAZ DE USUARIO
# ===========================

# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 Clasificador Spam vs No Spam</h1>
    <p>Regresión Logística + TF-IDF | Clasificación binaria de mensajes SMS</p>
</div>
""", unsafe_allow_html=True)

# Sidebar con info del modelo
with st.sidebar:
    st.markdown("### 📊 Info del Modelo")
    st.markdown("""
    - **Algoritmo**: Regresión Logística (L2)
    - **Vectorización**: TF-IDF (3000 features, n-gramas 1-2)
    - **Preprocesamiento**: Limpieza + Stemming + Stopwords
    - **Entrenamiento**: 4,459 mensajes (80%)
    - **Test**: 1,115 mensajes (20%)
    """)
    
    try:
        with open("models/model_metadata.json", "r") as f:
            import json
            meta = json.load(f)
        st.markdown("### 📈 Métricas (Test)")
        m = meta.get('metrics', {})
        st.metric("Accuracy", f"{m.get('accuracy', 0):.2%}")
        st.metric("Precision (Spam)", f"{m.get('precision', 0):.2%}")
        st.metric("Recall (Spam)", f"{m.get('recall', 0):.2%}")
        st.metric("F1-Score (Spam)", f"{m.get('f1', 0):.2%}")
        st.metric("ROC-AUC", f"{m.get('roc_auc', 0):.4f}")
    except:
        pass
    
    st.markdown("---")
    st.markdown("📁 **Repositorio**: [GitHub](https://github.com/022101096/REGRESION_LOGISTICA)")

# Área principal
st.markdown("### ✍️ Ingresa un mensaje para clasificar")

# Input de texto
user_input = st.text_area(
    "Mensaje SMS:",
    height=150,
    placeholder="Ej: FREE entry in a weekly comp to win FA Cup final tickets! Text FA to 87121...",
    help="Escribe o pega cualquier mensaje de texto. El modelo detectará si es spam o no spam."
)

# Botón de clasificar
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    classify_btn = st.button("🔍 Clasificar Mensaje", type="primary")

# Resultados
if classify_btn and user_input.strip():
    with st.spinner("Analizando mensaje..."):
        # Preprocesar
        clean = clean_text(user_input)
        X = vectorizer.transform([clean])
        X = scaler.transform(X)
        
        # Predicción
        prob_spam = model.predict_proba(X)[0, 1]
        pred = model.predict(X)[0]
        label = "SPAM" if pred == 1 else "NO SPAM (HAM)"
        
        # Explicabilidad
        explanation = get_explanation(vectorizer, model, scaler, user_input, top_n=5)
    
    # Mostrar resultado principal
    st.markdown("---")
    
    if pred == 1:
        st.markdown(f"""
        <div class="metric-card spam-alert">
            <h2 style="color: #E94F37; margin: 0;">🔴 PREDICCIÓN: {label}</h2>
            <h3 style="margin: 0.5rem 0;">Probabilidad de Spam: {prob_spam:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card ham-alert">
            <h2 style="color: #2E86AB; margin: 0;">🟢 PREDICCIÓN: {label}</h2>
            <h3 style="margin: 0.5rem 0;">Probabilidad de Spam: {prob_spam:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Barra de probabilidad visual
    st.markdown("#### 📊 Nivel de Confianza")
    progress_color = "#E94F37" if pred == 1 else "#2E86AB"
    st.markdown(f"""
    <div style="background: #e9ecef; border-radius: 10px; height: 30px; overflow: hidden;">
        <div style="width: {prob_spam*100}%; background: {progress_color}; height: 100%; 
                    display: flex; align-items: center; justify-content: flex-end; 
                    padding-right: 10px; color: white; font-weight: bold;
                    transition: width 0.5s ease;">
            {prob_spam:.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Explicabilidad
    st.markdown("---")
    st.markdown("### 🔍 ¿Por qué esta predicción?")
    st.caption("Las siguientes palabras del mensaje contribuyeron a la decisión (basado en coeficientes del modelo × valor TF-IDF):")
    
    col_spam, col_ham = st.columns(2)
    
    with col_spam:
        st.markdown("**🚨 Palabras que indican SPAM**")
        if explanation['spam']:
            for item in explanation['spam']:
                st.markdown(
                    f"<span class='feature-tag feature-spam'>{item['feature']} "
                    f"(+{item['contribution']:.3f})</span>",
                    unsafe_allow_html=True
                )
        else:
            st.caption("Ninguna palabra fuerte detectada")
    
    with col_ham:
        st.markdown("**✅ Palabras que indican NO SPAM**")
        if explanation['ham']:
            for item in explanation['ham']:
                st.markdown(
                    f"<span class='feature-tag feature-ham'>{item['feature']} "
                    f"({item['contribution']:.3f})</span>",
                    unsafe_allow_html=True
                )
        else:
            st.caption("Ninguna palabra fuerte detectada")
    
    # Texto limpio (debug)
    with st.expander("🔧 Ver texto preprocesado (debug)"):
        st.code(clean)

elif classify_btn and not user_input.strip():
    st.warning("⚠️ Por favor ingresa un mensaje antes de clasificar.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.85rem;">
    <p>Proyecto Académico - Taller Regresión Logística | 
    <a href="https://github.com/022101096/REGRESION_LOGISTICA" target="_blank">Ver código en GitHub</a></p>
    <p>Desplegado en Streamlit Community Cloud</p>
</div>
""", unsafe_allow_html=True)