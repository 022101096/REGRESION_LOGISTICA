# Clasificación Spam vs No Spam - Regresión Logística

Proyecto de clasificación binaria para detectar mensajes SMS spam usando Regresión Logística con TF-IDF.

## 📋 Descripción

Este proyecto implementa el ciclo completo de Machine Learning:
- **Análisis exploratorio** de datos (EDA)
- **Preprocesamiento** de texto con TF-IDF
- **Modelado** con Regresión Logística
- **Evaluación** con métricas estándar (Precision, Recall, F1, ROC-AUC)
- **Despliegue** de aplicación web interactiva con Streamlit

## 📊 Dataset

- **Fuente**: SMS Spam Collection (Kaggle / UCI)
- **Tamaño**: 5,574 mensajes
- **Clases**: `ham` (no spam) / `spam`
- **Balance**: ~74% ham, ~26% spam

## 🏗️ Estructura del Proyecto

```
├── archive/                 # Dataset original
├── data/
│   ├── raw/                 # Datos crudos (inmutables)
│   └── processed/           # Datos procesados + vectorizador
├── models/                  # Modelo entrenado + metadata
├── notebooks/
│   ├── 01_eda.ipynb         # Análisis exploratorio
│   ├── 02_preprocessing_modeling.ipynb  # Preprocesamiento + Entrenamiento
│   └── 03_evaluation_persistence.ipynb  # Evaluación + Persistencia
├── app/
│   ├── streamlit_app.py     # Aplicación web
│   └── requirements.txt     # Dependencias para deploy
├── reports/
│   ├── figures/             # Gráficos para el informe
│   └── informe_final.pdf    # Informe final (entregable)
├── src/
│   ├── preprocessing.py     # Funciones de preprocesamiento
│   ├── modeling.py          # Funciones de modelado
│   └── evaluation.py        # Funciones de evaluación
├── requirements.txt         # Dependencias del proyecto
└── .gitignore
```

## 🚀 Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/022101096/REGRESION_LOGISTICA.git
cd REGRESION_LOGISTICA

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Descargar stopwords NLTK
python -c "import nltk; nltk.download('stopwords')"
```

## 📓 Ejecutar Notebooks

Ejecutar en orden:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preprocessing_modeling.ipynb`
3. `notebooks/03_evaluation_persistence.ipynb`

## 🌐 Ejecutar App Localmente

```bash
streamlit run app/streamlit_app.py
```
Se abrirá en `http://localhost:8501`

## ☁️ Despliegue en Streamlit Cloud

1. Conectar repositorio en [share.streamlit.io](https://share.streamlit.io)
2. Seleccionar branch `main`, archivo `app/streamlit_app.py`
3. Deploy automático → URL pública

## 📈 Resultados Esperados

| Métrica | Valor Objetivo |
|---------|----------------|
| Accuracy | > 0.95 |
| Precision (Spam) | > 0.90 |
| Recall (Spam) | > 0.90 |
| F1-Score (Spam) | > 0.90 |
| ROC-AUC | > 0.98 |

## 👥 Equipo

- Integrante 1 - Código: XXXX
- Integrante 2 - Código: XXXX
- Integrante 3 - Código: XXXX
- Integrante 4 - Código: XXXX

## 📄 Licencia

MIT License