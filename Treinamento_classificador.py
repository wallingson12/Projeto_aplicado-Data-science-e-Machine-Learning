import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import joblib

# ---------- CONFIGURAÇÕES ----------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
FOLDER_PATH = "/home/wallingson/PycharmProjects/Projeto_aplicado/Treino"
DPI = 300
LANG = "por"
MODEL_FILENAME = "/home/wallingson/PycharmProjects/Projeto_aplicado/modelo_documentos.mkl"

def preprocess_image_for_ocr(pil_image):
    img = np.array(pil_image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = cv2.medianBlur(img_bin, 3)
    kernel = np.ones((2,2), np.uint8)
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)
    return Image.fromarray(img_bin)

def extract_text_from_pdf(pdf_path):
    try:
        # Converte APENAS a primeira página
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

def treinar_modelo():
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
        return None

    print("✅ Dataset criado com exemplos:")
    print(df.head())

    print("\n🤖 Treinando modelo de Machine Learning...")
    X = df["texto"]
    y = df["classe"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))
    modelo.fit(X_train, y_train)

    print("\n📈 Avaliando o modelo:")
    y_pred = modelo.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(modelo, MODEL_FILENAME)
    print(f"\n💾 Modelo salvo em '{MODEL_FILENAME}'")

if __name__ == "__main__":
    treinar_modelo()