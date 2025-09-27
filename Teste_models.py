import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import pandas as pd

# ---------- CONFIGURAÇÕES ----------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
FOLDER_PATH = "/home/wallingson/PycharmProjects/Projeto_aplicado/Treino"
DPI = 300
LANG = "por"

# ---------- PRÉ-PROCESSAMENTO DE IMAGEM ----------
def preprocess_image_for_ocr(pil_image):
    img = np.array(pil_image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = cv2.medianBlur(img_bin, 3)
    kernel = np.ones((2, 2), np.uint8)
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)
    return Image.fromarray(img_bin)

# ---------- EXTRAÇÃO DE TEXTO ----------
def extract_text_from_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=DPI, first_page=1, last_page=1)
        full_text = ""
        for image in images:
            img_pre = preprocess_image_for_ocr(image)
            text = pytesseract.image_to_string(img_pre, lang=LANG)
            full_text += text + "\n"
        return full_text
    except Exception as e:
        print(f"Erro na extração OCR de {pdf_path}: {e}")
        return ""

def preprocess_text(text):
    return text.lower().strip()

# ---------- LISTAGEM DE PDFs E ROTULOS ----------
def list_pdfs_with_labels(base_folder):
    pdfs_com_labels = []
    for root, dirs, files in os.walk(base_folder):
        classe = os.path.relpath(root, base_folder).split(os.sep)[0]
        if classe == ".":
            continue
        for file in files:
            if file.lower().endswith(".pdf"):
                caminho_pdf = os.path.join(root, file)
                pdfs_com_labels.append((caminho_pdf, classe))
    return pdfs_com_labels

# ---------- TREINO E AVALIAÇÃO DE MÚLTIPLOS MODELOS ----------
def treinar_e_comparar_modelos():
    print("📚 Extraindo textos para treino...\n")
    dados = []
    pdfs_rotulados = list_pdfs_with_labels(FOLDER_PATH)

    for caminho_pdf, classe in pdfs_rotulados:
        texto = extract_text_from_pdf(caminho_pdf)
        texto = preprocess_text(texto)
        if texto:
            print(f"✅ Texto extraído de {os.path.relpath(caminho_pdf, FOLDER_PATH)} (300 chars):\n{texto[:300]}\n")
            dados.append({"texto": texto, "classe": classe})
        else:
            print(f"⚠️ Nenhum texto extraído de {os.path.relpath(caminho_pdf, FOLDER_PATH)}")

    df = pd.DataFrame(dados)
    if df.empty:
        print("❌ Dataset vazio. Verifique seus PDFs e a extração de texto.")
        return

    print("✅ Dataset criado com exemplos:")
    print(df.head())

    X = df["texto"]
    y = df["classe"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelos = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Linear SVC": LinearSVC(),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    resultados = []

    print("\n🤖 Comparando modelos de classificação...\n")
    for nome, modelo in modelos.items():
        print(f"🔹 Modelo: {nome}")
        pipeline = make_pipeline(TfidfVectorizer(), modelo)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        resultados.append({
            "Modelo": nome,
            "Accuracy": report["accuracy"],
            "Precision (macro)": report["macro avg"]["precision"],
            "Recall (macro)": report["macro avg"]["recall"],
            "F1-score (macro)": report["macro avg"]["f1-score"],
            "Precision (weighted)": report["weighted avg"]["precision"],
            "Recall (weighted)": report["weighted avg"]["recall"],
            "F1-score (weighted)": report["weighted avg"]["f1-score"],
        })

        print(classification_report(y_test, y_pred, zero_division=0))
        print("-" * 60)

    df_resultados = pd.DataFrame(resultados)
    print("\n📊 Comparação final entre modelos:\n")
    print(df_resultados.sort_values(by="F1-score (weighted)", ascending=False).round(3))

# ---------- EXECUÇÃO ----------
if __name__ == "__main__":
    treinar_e_comparar_modelos()
