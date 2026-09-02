import sys
sys.path.append('D:/2026UAC/IA/Aplicacion-regresion')

import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation import (
    evaluate_model, print_metrics, plot_confusion_matrix,
    plot_roc_curve, plot_pr_curve, plot_coefficients,
    plot_class_distribution, save_metrics, threshold_analysis
)
from src.modeling import load_model, get_feature_importance

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(['#2E86AB', '#E94F37'])

# Cargar modelo y datos
model, vectorizer, scaler, metadata = load_model()
X_test = joblib.load('D:/2026UAC/IA/Aplicacion-regresion/data/processed/X_test.pkl')
y_test = joblib.load('D:/2026UAC/IA/Aplicacion-regresion/data/processed/y_test.pkl')

print(f'Modelo: {metadata["model_type"]}')
print(f'Features: {metadata["n_features"]}')
print(f'Test set: {X_test.shape[0]} muestras')
print(f'Distribucion test - Spam: {y_test.mean():.2%}')

# Evaluar
metrics, y_pred, y_proba, cm = evaluate_model(model, X_test, y_test, threshold=0.5)
print_metrics(metrics)

# Gráficos
plot_confusion_matrix(cm, save_path='D:/2026UAC/IA/Aplicacion-regresion/reports/figures/03_confusion_matrix.png')
plot_roc_curve(y_test, y_proba, save_path='D:/2026UAC/IA/Aplicacion-regresion/reports/figures/03_roc_curve.png')
plot_pr_curve(y_test, y_proba, save_path='D:/2026UAC/IA/Aplicacion-regresion/reports/figures/03_pr_curve.png')

# Análisis de umbral
best_threshold, best_f1 = threshold_analysis(y_test, y_proba)
print(f'\nMejor threshold: {best_threshold:.3f} (F1 = {best_f1:.4f})')

metrics_opt, _, _, _ = evaluate_model(model, X_test, y_test, threshold=best_threshold)
print_metrics(metrics_opt)

# Coeficientes
feature_importance = get_feature_importance(model, vectorizer, top_n=20)
plot_coefficients(feature_importance, top_n=15, save_path='D:/2026UAC/IA/Aplicacion-regresion/reports/figures/03_coefficients.png')

# Análisis de errores
df = pd.read_csv('D:/2026UAC/IA/Aplicacion-regresion/data/raw/spam.csv', encoding='latin-1')
df = df.iloc[:, :2]
df.columns = ['label', 'message']
df = df.drop_duplicates().reset_index(drop=True)
df['clean'] = df['message'].astype(str)

from sklearn.model_selection import train_test_split
_, X_test_text, _, _ = train_test_split(
    df['clean'], (df['label'] == 'spam').astype(int),
    test_size=0.2, random_state=42, stratify=(df['label'] == 'spam').astype(int)
)

errors_df = pd.DataFrame({
    'message': X_test_text.values,
    'true_label': y_test,
    'pred_label': y_pred,
    'prob_spam': y_proba
})

fp = errors_df[(errors_df['true_label'] == 0) & (errors_df['pred_label'] == 1)]
print(f'\nFalsos Positivos (Ham -> Spam): {len(fp)}')
if len(fp) > 0:
    print('Ejemplos:')
    for _, row in fp.head(3).iterrows():
        print(f"  P(Spam)={row['prob_spam']:.3f} | {row['message'][:100]}...")

fn = errors_df[(errors_df['true_label'] == 1) & (errors_df['pred_label'] == 0)]
print(f'\nFalsos Negativos (Spam -> Ham): {len(fn)}')
if len(fn) > 0:
    print('Ejemplos:')
    for _, row in fn.head(3).iterrows():
        print(f"  P(Spam)={row['prob_spam']:.3f} | {row['message'][:100]}...")

# Tabla métricas
metrics_table = pd.DataFrame({
    'Metrica': ['Accuracy', 'Precision (Spam)', 'Recall (Spam)', 'F1-Score (Spam)', 'ROC-AUC', 'PR-AUC'],
    'Valor': [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['roc_auc'],
        metrics['pr_auc']
    ]
})
metrics_table['Valor'] = metrics_table['Valor'].apply(lambda x: f'{x:.4f}')
print('\nTabla de metricas:')
print(metrics_table.to_string(index=False))
metrics_table.to_csv('D:/2026UAC/IA/Aplicacion-regresion/reports/figures/metrics_table.csv', index=False)

# Guardar modelo final con metricas reales
from src.modeling import save_model
final_metrics = {
    'accuracy': float(metrics['accuracy']),
    'precision': float(metrics['precision']),
    'recall': float(metrics['recall']),
    'f1': float(metrics['f1']),
    'roc_auc': float(metrics['roc_auc']),
    'pr_auc': float(metrics['pr_auc']),
    'threshold_used': 0.5,
    'optimal_threshold': float(best_threshold),
    'optimal_f1': float(best_f1)
}

save_model(model, vectorizer, scaler, final_metrics)
save_metrics(final_metrics, 'D:/2026UAC/IA/Aplicacion-regresion/models/metrics.json')

print('\nModelo final guardado con metricas reales:')
print(json.dumps(final_metrics, indent=2))

# Prueba rapida
from src.modeling import predict_with_explanation
test_msg = "FREE entry in a weekly comp to win FA Cup final tickets! Text FA to 87121"
result = predict_with_explanation(model, vectorizer, scaler, test_msg)
print(f'\nPrueba: "{test_msg[:50]}..."')
print(f'Prediccion: {result["prediction"]}')
print(f'Probabilidad Spam: {result["probability_spam"]:.4f}')
print(f'Explicacion Spam: {result["explanation"]["spam"][:3]}')