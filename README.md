# Classificador de PDFs com OCR e Machine Learning

Este projeto aplica técnicas de **Visão Computacional**, **Processamento de Linguagem Natural (NLP)** e **Machine Learning** para classificar documentos PDF em diferentes categorias com base no texto extraído via OCR.

## 🧠 Tecnologias utilizadas
- Python
- OpenCV
- Pytesseract
- pdf2image
- Scikit-learn
- Pandas
- Numpy

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

2. Execute o script:
python src/classificacao_pdfs.py

📊 Saída

O script extrai o texto dos PDFs via OCR, treina múltiplos modelos e compara os resultados de classificação por métricas como accuracy, f1-score, precision, etc.

🧪 Resultado esperado

Uma tabela comparativa entre os modelos:

Modelo	Accuracy	F1-score
Logistic Regression	0.85	0.84
Naive Bayes	0.82	0.81
