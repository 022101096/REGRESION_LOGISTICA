import sys
sys.path.append('D:/2026UAC/IA/Aplicacion-regresion')

import pandas as pd
import numpy as np
import joblib
from scipy.sparse import vstack

from src.preprocessing import prepare_data, save_processed_data
from src.modeling import train_logistic_regression, get_feature_importance, cross_validate_model, save_model

# Cargar datos
df = pd.read_csv('D:/2026UAC/IA/Aplicacion-regresion/data/raw/spam.csv', encoding='latin-1')
df = df.iloc[:, :2]
df.columns = ['label', 'message']
df = df.drop_duplicates().reset_index(drop=True)

print(f'Dataset cargado: {df.shape}')
print(df['label'].value_counts())

# Preprocesamiento
X_train, X_test, y_train, y_test, vectorizer, scaler = prepare_data(
    df,
    text_col='message',
    label_col='label',
    test_size=0.2,
    random_state=42,
    max_features=3000
)

# Guardar artefactos
save_processed_data(X_train, X_test, y_train, y_test, vectorizer, scaler)

# Entrenar modelo
model = train_logistic_regression(
    X_train, y_train,
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=2000,
    class_weight='balanced',
    random_state=42
)

# Validación cruzada
X_full = vstack([X_train, X_test])
y_full = np.concatenate([y_train, y_test])
cv_scores = cross_validate_model(model, X_full, y_full, cv=5)
print(f'\nF1 por fold: {cv_scores}')
print(f'Media: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# Análisis de coeficientes
feature_importance = get_feature_importance(model, vectorizer, top_n=20)

print('\n=== Top 20 Features -> SPAM ===')
spam_feats = feature_importance[feature_importance['class'] == 'spam']
print(spam_feats.to_string(index=False))

print('\n=== Top 20 Features -> HAM ===')
ham_feats = feature_importance[feature_importance['class'] == 'ham']
print(ham_feats.to_string(index=False))

# Gráfico coeficientes
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
top_n = 15
spam_df = spam_feats.head(top_n)
ham_df = ham_feats.head(top_n)

ax1.barh(range(len(spam_df)), spam_df['coefficient'], color='#E94F37', edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(len(spam_df)))
ax1.set_yticklabels(spam_df['feature'])
ax1.set_xlabel('Coeficiente')
ax1.set_title(f'Top {top_n} Features que indican SPAM', fontweight='bold', fontsize=12)
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

ax2.barh(range(len(ham_df)), ham_df['coefficient'], color='#2E86AB', edgecolor='black', linewidth=0.5)
ax2.set_yticks(range(len(ham_df)))
ax2.set_yticklabels(ham_df['feature'])
ax2.set_xlabel('Coeficiente')
ax2.set_title(f'Top {top_n} Features que indican HAM', fontweight='bold', fontsize=12)
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

plt.suptitle('Coeficientes de Regresión Logística', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/2026UAC/IA/Aplicacion-regresion/reports/figures/02_coefficients.png', dpi=300, bbox_inches='tight')
plt.close()

# Guardar modelo (métricas placeholder)
metrics_placeholder = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'roc_auc': 0.0}
save_model(model, vectorizer, scaler, metrics_placeholder)

print('\n✅ Preprocesamiento y modelado completado. Modelo guardado en models/')