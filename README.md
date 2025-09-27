# Classificador de PDFs com OCR e Machine Learning

Este projeto foi desenvolvido para a disciplina de Projeto Aplicado na Pós-Graduação em Data Science.
Utiliza **OCR (pytesseract)** para extrair texto de PDFs e aplica **classificadores de Machine Learning** para classificar automaticamente os documentos.

## 🧠 Tecnologias utilizadas
- Python
- OpenCV
- Pytesseract
- pdf2image
- Scikit-learn
- Pandas
- Numpy

⚠️ Requisitos

Tesseract OCR instalado no sistema

## 📁 Estrutura esperada
A pasta `Treino/` deve conter subpastas com os nomes das classes, e dentro delas os arquivos PDF.

Exemplo:

Treino/
├── juridico/
│ └── contrato1.pdf
└── financeiro/
└── relatorio1.pdf


## 🚀 Como executar

1. Instale os requisitos:

```bash
pip install -r requirements.txt

2. Para treinar o modelo:
python Treinamento_classificador.py

3. Para testar múltiplos modelos:
python teste_models.py

4. Para classificar novos PDFs:
python classificador.py

📊 Saída

O script extrai o texto dos PDFs via OCR, treina múltiplos modelos e compara os resultados de classificação por métricas como accuracy, f1-score, precision, etc.

🧪 Resultado esperado

Uma tabela comparativa entre os modelos:

Modelo	                |Accuracy|F1-score
Logistic Regression	    |0.85    |0.84
Naive Bayes	            |0.82    |0.81

Métricas avaliadas

Durante a avaliação dos modelos, são usadas as seguintes métricas:
