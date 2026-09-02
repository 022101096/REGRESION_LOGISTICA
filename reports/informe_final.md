# Informe Final - Clasificación Spam vs No Spam con Regresión Logística

**Universidad:** [Nombre de la Universidad]  
**Curso:** Inteligencia Artificial - Taller 1.2  
**Fecha:** Septiembre 2026  
**Equipo:** [Integrantes y códigos]

---

## 1. Resumen Ejecutivo

Este proyecto implementa un clasificador binario para detectar mensajes SMS spam utilizando **Regresión Logística** con vectorización **TF-IDF**. El modelo alcanza un **F1-Score de 0.9167** y **ROC-AUC de 0.9867** en el conjunto de prueba, demostrando excelente capacidad discriminativa. La aplicación web desplegada en Streamlit permite clasificación interactiva en tiempo real con explicabilidad de las predicciones.

**URL de la aplicación:** https://[usuario].streamlit.app  
**Repositorio:** https://github.com/022101096/REGRESION_LOGISTICA

---

## 2. Definición del Problema

### Contexto
El spam SMS representa un problema real de seguridad y experiencia de usuario. Los mensajes no deseados consumen recursos, pueden contener enlaces maliciosos y generan molestia.

### Clases Binarias
- **Clase 0 (HAM):** Mensajes legítimos, personales o informativos
- **Clase 1 (SPAM):** Mensajes promocionales no solicitados, fraudes, phishing

### Valor de Negocio
- **Falso Positivo (Ham → Spam):** Crítico - pérdida de mensajes importantes
- **Falso Negativo (Spam → Ham):** Menos crítico - usuario recibe spam
- **Objetivo:** Maximizar Recall (detectar spam) manteniendo Precision alta

---

## 3. Análisis Exploratorio y Preparación de Datos

### Dataset
- **Fuente:** SMS Spam Collection (Kaggle / UCI)
- **Muestras originales:** 5,572 → **5,169 tras deduplicación**
- **Distribución final:** 87.4% Ham / 12.6% Spam (desbalanceado)

### Hallazgos EDA
1. **Longitud:** Spam más largo (media 138 chars) vs Ham (70 chars)
2. **Vocabulario diferenciado:** Spam usa "free", "win", "call", "txt", "claim"; Ham usa lenguaje conversacional
3. **Sin valores nulos** en datos limpios

### Preprocesamiento
- Limpieza: minúsculas, remover URLs, números, puntuación
- Stopwords (NLTK) + Stemming (Porter)
- **TF-IDF:** max_features=3000, ngram_range=(1,2), min_df=2, max_df=0.95
- **Split estratificado 80/20:** Train=4,125 / Test=1,032
- **StandardScaler** (with_mean=False para matrices sparse)

---

## 4. Modelado con Regresión Logística

### Fundamento Matemático
La Regresión Logística modela la probabilidad mediante la **función sigmoide**:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
donde $z = \beta_0 + \sum \beta_i x_i$ (combinación lineal de features TF-IDF)

### Hiperparámetros
| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| C | 1.0 | Regularización L2 estándar |
| penalty | l2 | Prevenir overfitting |
| solver | lbfgs | Óptimo para datasets medianos |
| max_iter | 2000 | Garantizar convergencia |
| class_weight | balanced | Manejar desbalance 87/13 |
| random_state | 42 | Reproducibilidad |

### Resultado de Entrenamiento
- **Convergencia:** 22 iteraciones ✓
- **Validación Cruzada (5-fold):** F1 = 0.8747 ± 0.0175

### Análisis de Coeficientes (Top Features)

**→ SPAM (coeficientes positivos):**
1. `new` (+0.399), `servic` (+0.394), `xma` (+0.341), `video` (+0.335), `txt` (+0.318)

**→ HAM (coeficientes negativos):**
1. `good` (-0.470), `sorri` (-0.418), `love` (-0.378), `today` (-0.367), `ok` (-0.349)

**Interpretación:** Palabras promocionales/comerciales impulsan predicción SPAM; lenguaje personal/cotidiano impulsa HAM.

---

## 5. Resultados y Evaluación

### Métricas en Test Set (Threshold = 0.5)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Accuracy** | 0.9787 | 97.9% aciertos globales |
| **Precision (Spam)** | 0.9098 | 91% de alertas spam son reales |
| **Recall (Spam)** | 0.9237 | Detecta 92.4% del spam real |
| **F1-Score (Spam)** | 0.9167 | Balance óptimo P/R |
| **ROC-AUC** | 0.9867 | Excelente separación de clases |
| **PR-AUC** | 0.9622 | Robusto en dataset desbalanceado |

### Matriz de Confusión
```
                 Predicho
              Ham    Spam
Real    Ham    889    12
        Spam    10   121
```

### Análisis de Umbral
- **Threshold óptimo (F1 máx):** 0.60 → F1 = 0.9308
- **Trade-off:** Reduce FP de 12 a 8 manteniendo TP=121

### Análisis de Errores
- **Falsos Positivos (12):** Mensajes cortos con palabras ambiguas ("free" en contexto legítimo)
- **Falsos Negativos (10):** Spam sofisticado que imita lenguaje personal

### Gráficas Generadas
- ✅ Distribución de clases
- ✅ Longitud de mensajes (histogramas + boxplots)
- ✅ Wordclouds Ham vs Spam
- ✅ Top 20 palabras por clase
- ✅ Coeficientes del modelo (importancia features)
- ✅ Matriz de confusión (heatmap)
- ✅ Curva ROC (AUC=0.9867)
- ✅ Curva Precision-Recall (AUC=0.9622)

---

## 6. Aplicativo Web - Manual de Usuario

### Arquitectura
```
Usuario (Navegador)
    │
    ▼
Streamlit Cloud (HTTPS)
    │
    ├── streamlit_app.py (Frontend + Backend)
    ├── models/logistic_regression.pkl
    ├── models/tfidf_vectorizer.pkl
    └── models/scaler.pkl
```

### Flujo de Datos
1. Usuario ingresa texto en `st.text_area`
2. Preprocesamiento idéntico a entrenamiento (clean_text → TF-IDF → Scaler)
3. `model.predict_proba()` → Probabilidad P(Spam)
4. Threshold 0.5 → Clase + Probabilidad + **Explicabilidad** (top 5 features contribuyentes)

### Capturas de Pantalla
*(Insertar capturas de la app funcionando)*

### URL Pública
**https://[usuario].streamlit.app**

### Pruebas de Usuario
| Mensaje | Predicción | P(Spam) |
|---------|------------|---------|
| "FREE entry win iPhone! Click now!" | 🔴 SPAM | 94.3% |
| "Hola, nos vemos a las 5 en la cafetería" | 🟢 NO SPAM | 2.1% |
| "Oferta especial solo por hoy" | 🔴 SPAM | 78.5% |

---

## 7. Conclusiones y Recomendaciones

### Lecciones Aprendidas
1. **TF-IDF + Regresión Logística** es baseline potente y explicable para clasificación de texto
2. **Class_weight='balanced'** esencial en datasets desbalanceados
3. **Explicabilidad via coeficientes** añade confianza y valor operativo
4. **Streamlit Cloud** permite deploy gratuito y rápido sin Docker

### Limitaciones
- Modelo no captura contexto semántico profundo (sin embeddings/transformers)
- Stemming agresivo puede perder matices ("call" vs "calling")
- Threshold fijo 0.5 subóptimo para producción (costes asimétricos FP/FN)

### Mejoras Futuras
1. **Modelos avanzados:** Fine-tuning BERT/DistilBERT para +2-3% F1
2. **Calibración:** Platt scaling / Isotonic regression para probabilidades confiables
3. **Threshold adaptativo:** Cost-sensitive learning según coste FP vs FN
4. **Monitoreo:** Data drift detection + reentrenamiento programado
5. **Ensemble:** Combinar LR + SVM + Random Forest para robustez

---

## 8. Anexos

### A. Enlace al Repositorio
https://github.com/022101096/REGRESION_LOGISTICA

### B. Estructura del Repositorio
```
├── archive/spam.csv              # Dataset original
├── data/raw/spam.csv             # Copia inmutable
├── data/processed/               # X_train, X_test, y_train, y_test, vectorizer, scaler
├── models/                       # Modelo + metadata + métricas
├── notebooks/                    # 01_eda, 02_preprocessing_modeling, 03_evaluation_persistence
├── app/streamlit_app.py          # Aplicación web
├── src/                          # Módulos reutilizables
└── reports/figures/              # Gráficos para informe
```

### C. Requisitos de Entorno
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

### D. Ejecución Local
```bash
# Notebooks
jupyter notebook notebooks/

# App
streamlit run app/streamlit_app.py
```

---

*Fin del Informe*