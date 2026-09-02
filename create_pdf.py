from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, 'Informe Final - Clasificacion Spam vs No Spam', 0, 1, 'C')
        self.line(10, 12, 200, 12)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Pagina {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, num, title):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(46, 134, 171)  # #2E86AB
        self.cell(0, 10, f'{num}. {title}', 0, 1, 'L')
        self.ln(2)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(233, 79, 55)  # #E94F37
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(1)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bullet_point(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(50, 50, 50)
        self.cell(5, 5, '-', 0, 0)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def bold_text(self, text):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def add_table(self, headers, data, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        
        # Header
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(46, 134, 171)
        self.set_text_color(255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, 1, 0, 'C', True)
        self.ln()
        
        # Data
        self.set_font('Helvetica', '', 9)
        self.set_text_color(50, 50, 50)
        fill = False
        for row in data:
            if fill:
                self.set_fill_color(240, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), 1, 0, 'C', True)
            self.ln()
            fill = not fill
        self.ln(3)


def create_pdf():
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Portada
    pdf.ln(30)
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 15, 'Clasificacion Spam vs No Spam', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 16)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, 'Regresion Logistica + TF-IDF', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 8, 'Taller 1.2 - Modelos de Regresion Logistica', 0, 1, 'C')
    pdf.cell(0, 8, 'Practica Calificada: Clasificacion Binaria', 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, 'Integrantes: [Nombres y Codigos]', 0, 1, 'C')
    pdf.cell(0, 7, 'Fecha: Septiembre 2026', 0, 1, 'C')
    pdf.cell(0, 7, 'Universidad: [Nombre]', 0, 1, 'C')

    # Tabla de contenido
    pdf.add_page()
    pdf.chapter_title('', 'Tabla de Contenido')
    toc = [
        ('1', 'Resumen Ejecutivo'),
        ('2', 'Definicion del Problema'),
        ('3', 'Analisis Exploratorio y Preparacion de Datos'),
        ('4', 'Modelado con Regresion Logistica'),
        ('5', 'Resultados y Evaluacion'),
        ('6', 'Aplicativo Web - Manual de Usuario'),
        ('7', 'Conclusiones y Recomendaciones'),
        ('8', 'Anexos'),
    ]
    for num, title in toc:
        pdf.set_font('Helvetica', '', 11)
        pdf.cell(0, 8, f'{num}.  {title}', 0, 1, 'L')

    # 1. Resumen Ejecutivo
    pdf.add_page()
    pdf.chapter_title('1', 'Resumen Ejecutivo')
    pdf.body_text(
        'Este proyecto implementa un clasificador binario para detectar mensajes SMS spam '
        'utilizando Regresion Logistica con vectorizacion TF-IDF. El modelo alcanza un '
        'F1-Score de 0.9167 y ROC-AUC de 0.9867 en el conjunto de prueba, demostrando '
        'excelente capacidad discriminativa. La aplicacion web desplegada en Streamlit '
        'permite clasificacion interactiva en tiempo real con explicabilidad de las predicciones.'
    )
    pdf.bold_text('URL de la aplicacion: https://[usuario].streamlit.app')
    pdf.bold_text('Repositorio: https://github.com/022101096/REGRESION_LOGISTICA')

    # 2. Definicion del Problema
    pdf.chapter_title('2', 'Definicion del Problema')
    pdf.section_title('Contexto')
    pdf.body_text(
        'El spam SMS representa un problema real de seguridad y experiencia de usuario. '
        'Los mensajes no deseados consumen recursos, pueden contener enlaces maliciosos y generan molestia.'
    )
    pdf.section_title('Clases Binarias')
    pdf.bullet_point('Clase 0 (HAM): Mensajes legitimos, personales o informativos')
    pdf.bullet_point('Clase 1 (SPAM): Mensajes promocionales no solicitados, fraudes, phishing')
    pdf.section_title('Valor de Negocio')
    pdf.bullet_point('Falso Positivo (Ham -> Spam): Critico - perdida de mensajes importantes')
    pdf.bullet_point('Falso Negativo (Spam -> Ham): Menos critico - usuario recibe spam')
    pdf.bullet_point('Objetivo: Maximizar Recall (detectar spam) manteniendo Precision alta')

    # 3. EDA y Preparacion
    pdf.chapter_title('3', 'Analisis Exploratorio y Preparacion de Datos')
    pdf.section_title('Dataset')
    pdf.bullet_point('Fuente: SMS Spam Collection (Kaggle / UCI)')
    pdf.bullet_point('Muestras originales: 5,572 -> 5,169 tras deduplicacion')
    pdf.bullet_point('Distribucion final: 87.4% Ham / 12.6% Spam (desbalanceado)')
    pdf.section_title('Hallazgos EDA')
    pdf.bullet_point('Longitud: Spam mas largo (media 138 chars) vs Ham (70 chars)')
    pdf.bullet_point('Vocabulario diferenciado: Spam usa "free", "win", "call", "txt", "claim"')
    pdf.bullet_point('Sin valores nulos en datos limpios')
    pdf.section_title('Preprocesamiento')
    pdf.bullet_point('Limpieza: minusculas, remover URLs, numeros, puntuacion')
    pdf.bullet_point('Stopwords (NLTK) + Stemming (Porter)')
    pdf.bullet_point('TF-IDF: max_features=3000, ngram_range=(1,2), min_df=2, max_df=0.95')
    pdf.bullet_point('Split estratificado 80/20: Train=4,125 / Test=1,032')
    pdf.bullet_point('StandardScaler (with_mean=False para matrices sparse)')

    # 4. Modelado
    pdf.add_page()
    pdf.chapter_title('4', 'Modelado con Regresion Logistica')
    pdf.section_title('Fundamento Matematico')
    pdf.body_text(
        'La Regresion Logistica modela la probabilidad mediante la funcion sigmoide: '
        'sigma(z) = 1 / (1 + e^{-z}) donde z = beta_0 + sum(beta_i * x_i) '
        '(combinacion lineal de features TF-IDF).'
    )
    pdf.section_title('Hiperparametros')
    pdf.add_table(
        ['Parametro', 'Valor', 'Justificacion'],
        [
            ['C', '1.0', 'Regularizacion L2 estandar'],
            ['penalty', 'l2', 'Prevenir overfitting'],
            ['solver', 'lbfgs', 'Optimo para datasets medianos'],
            ['max_iter', '2000', 'Garantizar convergencia'],
            ['class_weight', 'balanced', 'Manejar desbalance 87/13'],
            ['random_state', '42', 'Reproducibilidad'],
        ],
        col_widths=[40, 30, 120]
    )
    pdf.section_title('Resultado de Entrenamiento')
    pdf.bullet_point('Convergencia: 22 iteraciones')
    pdf.bullet_point('Validacion Cruzada (5-fold): F1 = 0.8747 +/- 0.0175')
    pdf.section_title('Analisis de Coeficientes (Top Features)')
    pdf.bold_text('-> SPAM (coeficientes positivos):')
    pdf.bullet_point('new (+0.399), servic (+0.394), xma (+0.341), video (+0.335), txt (+0.318)')
    pdf.bold_text('-> HAM (coeficientes negativos):')
    pdf.bullet_point('good (-0.470), sorri (-0.418), love (-0.378), today (-0.367), ok (-0.349)')
    pdf.body_text(
        'Interpretacion: Palabras promocionales/comerciales impulsan prediccion SPAM; '
        'lenguaje personal/cotidiano impulsa HAM.'
    )

    # 5. Resultados
    pdf.add_page()
    pdf.chapter_title('5', 'Resultados y Evaluacion')
    pdf.section_title('Metricas en Test Set (Threshold = 0.5)')
    pdf.add_table(
        ['Metrica', 'Valor', 'Interpretacion'],
        [
            ['Accuracy', '0.9787', '97.9% aciertos globales'],
            ['Precision (Spam)', '0.9098', '91% de alertas spam son reales'],
            ['Recall (Spam)', '0.9237', 'Detecta 92.4% del spam real'],
            ['F1-Score (Spam)', '0.9167', 'Balance optimo P/R'],
            ['ROC-AUC', '0.9867', 'Excelente separacion de clases'],
            ['PR-AUC', '0.9622', 'Robusto en dataset desbalanceado'],
        ],
        col_widths=[45, 25, 120]
    )
    pdf.section_title('Matriz de Confusion')
    pdf.add_table(
        ['', 'Pred. Ham', 'Pred. Spam'],
        [
            ['Real Ham', '889', '12'],
            ['Real Spam', '10', '121'],
        ],
        col_widths=[50, 70, 70]
    )
    pdf.section_title('Analisis de Umbral')
    pdf.bullet_point('Threshold optimo (F1 max): 0.60 -> F1 = 0.9308')
    pdf.bullet_point('Trade-off: Reduce FP de 12 a 8 manteniendo TP=121')
    pdf.section_title('Analisis de Errores')
    pdf.bullet_point('Falsos Positivos (12): Mensajes cortos con palabras ambiguas')
    pdf.bullet_point('Falsos Negativos (10): Spam sofisticado que imita lenguaje personal')
    pdf.section_title('Graficas Generadas')
    figures = [
        'Distribucion de clases', 'Longitud de mensajes', 'Wordclouds Ham vs Spam',
        'Top 20 palabras por clase', 'Coeficientes del modelo', 'Matriz de confusion',
        'Curva ROC (AUC=0.9867)', 'Curva Precision-Recall (AUC=0.9622)'
    ]
    for f in figures:
        pdf.bullet_point(f)

    # 6. Aplicativo Web
    pdf.add_page()
    pdf.chapter_title('6', 'Aplicativo Web - Manual de Usuario')
    pdf.section_title('Arquitectura')
    pdf.body_text(
        'Usuario (Navegador) -> Streamlit Cloud (HTTPS) -> streamlit_app.py '
        '(Frontend + Backend) + modelos .pkl'
    )
    pdf.section_title('Flujo de Datos')
    pdf.bullet_point('1. Usuario ingresa texto en text_area')
    pdf.bullet_point('2. Preprocesamiento identico a entrenamiento (clean_text -> TF-IDF -> Scaler)')
    pdf.bullet_point('3. model.predict_proba() -> Probabilidad P(Spam)')
    pdf.bullet_point('4. Threshold 0.5 -> Clase + Probabilidad + Explicabilidad (top 5 features)')
    pdf.section_title('URL Publica')
    pdf.bold_text('https://[usuario].streamlit.app')
    pdf.section_title('Pruebas de Usuario')
    pdf.add_table(
        ['Mensaje', 'Prediccion', 'P(Spam)'],
        [
            ['"FREE entry win iPhone! Click now!"', 'SPAM', '94.3%'],
            ['"Hola, nos vemos a las 5"', 'NO SPAM', '2.1%'],
            ['"Oferta especial solo por hoy"', 'SPAM', '78.5%'],
        ],
        col_widths=[80, 35, 35]
    )

    # 7. Conclusiones
    pdf.add_page()
    pdf.chapter_title('7', 'Conclusiones y Recomendaciones')
    pdf.section_title('Lecciones Aprendidas')
    pdf.bullet_point('TF-IDF + Regresion Logistica es baseline potente y explicable para texto')
    pdf.bullet_point('class_weight=balanced esencial en datasets desbalanceados')
    pdf.bullet_point('Explicabilidad via coeficientes anade confianza y valor operativo')
    pdf.bullet_point('Streamlit Cloud permite deploy gratuito y rapido sin Docker')
    pdf.section_title('Limitaciones')
    pdf.bullet_point('No captura contexto semantico profundo (sin embeddings/transformers)')
    pdf.bullet_point('Stemming agresivo puede perder matices ("call" vs "calling")')
    pdf.bullet_point('Threshold fijo 0.5 suboptimo para produccion (costes asimetricos FP/FN)')
    pdf.section_title('Mejoras Futuras')
    pdf.bullet_point('Modelos avanzados: Fine-tuning BERT/DistilBERT para +2-3% F1')
    pdf.bullet_point('Calibracion: Platt scaling / Isotonic regression')
    pdf.bullet_point('Threshold adaptativo: Cost-sensitive learning')
    pdf.bullet_point('Monitoreo: Data drift detection + reentrenamiento programado')
    pdf.bullet_point('Ensemble: Combinar LR + SVM + Random Forest')

    # 8. Anexos
    pdf.chapter_title('8', 'Anexos')
    pdf.section_title('A. Enlace al Repositorio')
    pdf.body_text('https://github.com/022101096/REGRESION_LOGISTICA')
    pdf.section_title('B. Estructura del Repositorio')
    pdf.body_text(
        'archive/spam.csv\n'
        'data/raw/spam.csv\n'
        'data/processed/ (X_train, X_test, y_train, y_test, vectorizer, scaler)\n'
        'models/ (modelo + metadata + metricas)\n'
        'notebooks/ (01_eda, 02_preprocessing_modeling, 03_evaluation_persistence)\n'
        'app/streamlit_app.py\n'
        'src/ (modulos reutilizables)\n'
        'reports/figures/ (graficos para informe)'
    )
    pdf.section_title('C. Requisitos de Entorno')
    pdf.body_text(
        'pip install -r requirements.txt\n'
        'python -c "import nltk; nltk.download(\'stopwords\')"'
    )
    pdf.section_title('D. Ejecucion Local')
    pdf.body_text(
        '# Notebooks\n'
        'jupyter notebook notebooks/\n\n'
        '# App\n'
        'streamlit run app/streamlit_app.py'
    )

    # Guardar
    output_path = 'D:/2026UAC/IA/Aplicacion-regresion/reports/informe_final.pdf'
    pdf.output(output_path)
    print(f'PDF generado: {output_path}')

if __name__ == '__main__':
    create_pdf()