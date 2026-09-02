import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report
)
import joblib
import json


def evaluate_model(model, X_test, y_test, threshold: float = 0.5):
    """
    Evaluación completa del modelo.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Métricas principales
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba)
    }
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    })
    
    return metrics, y_pred, y_proba, cm


def print_metrics(metrics: dict):
    """
    Imprime métricas formateadas.
    """
    print("\n" + "="*50)
    print("MÉTRICAS DE EVALUACIÓN (Test)")
    print("="*50)
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1-Score:     {metrics['f1']:.4f}")
    print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:       {metrics['pr_auc']:.4f}")
    print("-"*50)
    print("Matriz de Confusión:")
    print(f"  TN: {metrics['true_negatives']:4d}  FP: {metrics['false_positives']:4d}")
    print(f"  FN: {metrics['false_negatives']:4d}  TP: {metrics['true_positives']:4d}")
    print("="*50)


def plot_confusion_matrix(cm, save_path: str = None):
    """
    Grafica matriz de confusión.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham (0)', 'Spam (1)'],
                yticklabels=['Ham (0)', 'Spam (1)'],
                cbar_kws={'label': 'Cantidad'})
    plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
    plt.ylabel('Real', fontsize=12)
    plt.xlabel('Predicho', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en {save_path}")
    plt.show()


def plot_roc_curve(y_test, y_proba, save_path: str = None):
    """
    Grafica curva ROC.
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#2E86AB', lw=2, 
             label=f'Regresión Logística (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Azar (AUC = 0.5)')
    plt.fill_between(fpr, tpr, alpha=0.1, color='#2E86AB')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    plt.title('Curva ROC - Clasificador Spam', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en {save_path}")
    plt.show()


def plot_pr_curve(y_test, y_proba, save_path: str = None):
    """
    Grafica curva Precision-Recall.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    # Baseline (proporción de positivos)
    baseline = y_test.mean()
    
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color='#E94F37', lw=2,
             label=f'Regresión Logística (PR-AUC = {pr_auc:.4f})')
    plt.axhline(y=baseline, color='gray', lw=1, linestyle='--', 
                label=f'Baseline (Spam = {baseline:.2%})')
    plt.fill_between(recall, precision, alpha=0.1, color='#E94F37')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensibilidad)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Curva Precision-Recall', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en {save_path}")
    plt.show()


def plot_coefficients(feature_importance_df, top_n: int = 15, save_path: str = None):
    """
    Grafica coeficientes top (spam vs ham).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Spam (coeficientes positivos)
    spam_df = feature_importance_df[feature_importance_df['class'] == 'spam'].head(top_n)
    colors_spam = ['#E94F37' if c > 0 else '#2E86AB' for c in spam_df['coefficient']]
    ax1.barh(range(len(spam_df)), spam_df['coefficient'], color=colors_spam, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(spam_df)))
    ax1.set_yticklabels(spam_df['feature'])
    ax1.set_xlabel('Coeficiente', fontsize=11)
    ax1.set_title(f'Top {top_n} Features → SPAM', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Ham (coeficientes negativos)
    ham_df = feature_importance_df[feature_importance_df['class'] == 'ham'].head(top_n)
    colors_ham = ['#E94F37' if c > 0 else '#2E86AB' for c in ham_df['coefficient']]
    ax2.barh(range(len(ham_df)), ham_df['coefficient'], color=colors_ham, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(ham_df)))
    ax2.set_yticklabels(ham_df['feature'])
    ax2.set_xlabel('Coeficiente', fontsize=11)
    ax2.set_title(f'Top {top_n} Features → HAM (No Spam)', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Coeficientes de Regresión Logística (Importancia de Features)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en {save_path}")
    plt.show()


def plot_class_distribution(y_train, y_test, save_path: str = None):
    """
    Grafica distribución de clases en train y test.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    for ax, y, title in [(ax1, y_train, 'Train'), (ax2, y_test, 'Test')]:
        counts = pd.Series(y).value_counts().sort_index()
        labels = ['Ham', 'Spam']
        colors = ['#2E86AB', '#E94F37']
        bars = ax.bar(labels, counts.values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(f'{title} ({len(y)} muestras)', fontweight='bold')
        ax.set_ylabel('Cantidad')
        # Porcentajes en barras
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'{count}\n({count/len(y):.1%})', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Distribución de Clases', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en {save_path}")
    plt.show()


def save_metrics(metrics: dict, output_path: str = 'models/metrics.json'):
    """
    Guarda métricas en JSON.
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas guardadas en {output_path}")


def threshold_analysis(y_test, y_proba):
    """
    Análisis de umbral óptimo (F1 máximo).
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    precisions = []
    recalls = []
    
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        f1_scores.append(f1_score(y_test, y_pred_t))
        precisions.append(precision_score(y_test, y_pred_t))
        recalls.append(recall_score(y_test, y_pred_t))
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, f1_scores, label='F1-Score', color='#2E86AB', lw=2)
    plt.plot(thresholds, precisions, label='Precision', color='#E94F37', lw=2)
    plt.plot(thresholds, recalls, label='Recall', color='#28A745', lw=2)
    plt.axvline(x=best_threshold, color='gray', linestyle='--', 
                label=f'Mejor F1 en threshold={best_threshold:.2f}')
    plt.axvline(x=0.5, color='black', linestyle=':', label='Threshold por defecto (0.5)')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Análisis de Umbral de Decisión', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return best_threshold, f1_scores[best_idx]