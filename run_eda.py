import sys
sys.path.append('D:/2026UAC/IA/Aplicacion-regresion')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(['#2E86AB', '#E94F37'])
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Cargar dataset
df = pd.read_csv('D:/2026UAC/IA/Aplicacion-regresion/data/raw/spam.csv', encoding='latin-1')
df = df.iloc[:, :2]
df.columns = ['label', 'message']

print(f'Shape: {df.shape}')
print(f'\nColumnas: {df.columns.tolist()}')
print(f'\nTipos de datos:\n{df.dtypes}')
print(f'\nPrimeras 5 filas:')
print(df.head())

# Info general
print(df.info())
print(f'\nValores nulos:\n{df.isnull().sum()}')

# Duplicados
duplicates = df.duplicated().sum()
print(f'\nFilas duplicadas: {duplicates}')
if duplicates > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f'Duplicados eliminados. Nueva shape: {df.shape}')

# Distribución de clases
class_counts = df['label'].value_counts()
class_pct = df['label'].value_counts(normalize=True) * 100
dist_df = pd.DataFrame({'Count': class_counts, 'Percentage': class_pct})
print(dist_df)

# Gráfico distribución
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
bars = ax1.bar(dist_df.index, dist_df['Count'], color=['#2E86AB', '#E94F37'], edgecolor='black', linewidth=0.5)
ax1.set_title('Distribución de Clases', fontweight='bold', fontsize=14)
ax1.set_ylabel('Cantidad de Mensajes')
for bar, count, pct in zip(bars, dist_df['Count'], dist_df['Percentage']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax2.pie(dist_df['Count'], labels=dist_df.index, autopct='%1.1f%%',
        colors=['#2E86AB', '#E94F37'], startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('Proporción de Clases', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('D:/2026UAC/IA/Aplicacion-regresion/reports/figures/01_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Longitud mensajes
df['char_length'] = df['message'].str.len()
df['word_count'] = df['message'].str.split().str.len()

print('=== Longitud en caracteres ===')
print(df.groupby('label')['char_length'].describe())
print('\n=== Conteo de palabras ===')
print(df.groupby('label')['word_count'].describe())

# Histogramas
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for label, color in [('ham', '#2E86AB'), ('spam', '#E94F37')]:
    data = df[df['label'] == label]['char_length']
    axes[0,0].hist(data, bins=50, alpha=0.6, label=label, color=color, edgecolor='black', linewidth=0.3)
axes[0,0].set_title('Longitud (caracteres) por Clase', fontweight='bold')
axes[0,0].set_xlabel('Caracteres')
axes[0,0].set_ylabel('Frecuencia')
axes[0,0].legend()
axes[0,0].set_xlim(0, 200)

df.boxplot(column='char_length', by='label', ax=axes[0,1], grid=False)
axes[0,1].set_title('Boxplot: Longitud en Caracteres', fontweight='bold')
axes[0,1].set_xlabel('Clase')
axes[0,1].set_ylabel('Caracteres')

for label, color in [('ham', '#2E86AB'), ('spam', '#E94F37')]:
    data = df[df['label'] == label]['word_count']
    axes[1,0].hist(data, bins=30, alpha=0.6, label=label, color=color, edgecolor='black', linewidth=0.3)
axes[1,0].set_title('Conteo de Palabras por Clase', fontweight='bold')
axes[1,0].set_xlabel('Palabras')
axes[1,0].set_ylabel('Frecuencia')
axes[1,0].legend()
axes[1,0].set_xlim(0, 50)

df.boxplot(column='word_count', by='label', ax=axes[1,1], grid=False)
axes[1,1].set_title('Boxplot: Conteo de Palabras', fontweight='bold')
axes[1,1].set_xlabel('Clase')
axes[1,1].set_ylabel('Palabras')

plt.tight_layout()
plt.savefig('D:/2026UAC/IA/Aplicacion-regresion/reports/figures/01_message_length.png', dpi=300, bbox_inches='tight')
plt.close()

# Wordclouds
ham_text = ' '.join(df[df['label'] == 'ham']['message'].astype(str))
spam_text = ' '.join(df[df['label'] == 'spam']['message'].astype(str))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
wc_ham = WordCloud(width=800, height=400, background_color='white',
                  colormap='Blues', max_words=100, contour_width=1, contour_color='steelblue').generate(ham_text)
ax1.imshow(wc_ham, interpolation='bilinear')
ax1.set_title('Wordcloud: HAM (No Spam)', fontweight='bold', fontsize=16)
ax1.axis('off')

wc_spam = WordCloud(width=800, height=400, background_color='white',
                   colormap='Reds', max_words=100, contour_width=1, contour_color='darkred').generate(spam_text)
ax2.imshow(wc_spam, interpolation='bilinear')
ax2.set_title('Wordcloud: SPAM', fontweight='bold', fontsize=16)
ax2.axis('off')

plt.tight_layout()
plt.savefig('D:/2026UAC/IA/Aplicacion-regresion/reports/figures/01_wordclouds.png', dpi=300, bbox_inches='tight')
plt.close()

# Top palabras
from collections import Counter
import re

def get_top_words(texts, n=20):
    all_words = []
    for text in texts:
        words = re.findall(r'\b[a-z]{2,}\b', text.lower())
        all_words.extend(words)
    return Counter(all_words).most_common(n)

ham_words = get_top_words(df[df['label'] == 'ham']['message'], 20)
spam_words = get_top_words(df[df['label'] == 'spam']['message'], 20)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
words, counts = zip(*ham_words)
ax1.barh(range(len(words)), counts, color='#2E86AB', edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(len(words)))
ax1.set_yticklabels(words)
ax1.set_title('Top 20 Palabras - HAM', fontweight='bold', fontsize=14)
ax1.set_xlabel('Frecuencia')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

words, counts = zip(*spam_words)
ax2.barh(range(len(words)), counts, color='#E94F37', edgecolor='black', linewidth=0.5)
ax2.set_yticks(range(len(words)))
ax2.set_yticklabels(words)
ax2.set_title('Top 20 Palabras - SPAM', fontweight='bold', fontsize=14)
ax2.set_xlabel('Frecuencia')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('D:/2026UAC/IA/Aplicacion-regresion/reports/figures/01_top_words.png', dpi=300, bbox_inches='tight')
plt.close()

print('\n✅ EDA completado. Gráficos guardados en reports/figures/')