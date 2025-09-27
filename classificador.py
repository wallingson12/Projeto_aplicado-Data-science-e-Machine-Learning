import os
import time
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import joblib

# ---------- CONFIGURAÇÕES ----------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
FOLDER_PATH = "/home/wallingson/PycharmProjects/Projeto_aplicado/Teste"
DPI = 300  # Reduzido para performance no Colab
LANG = "por"
MODEL_FILENAME = "/home/wallingson/PycharmProjects/Projeto_aplicado/modelo_documentos.mkl"

def preprocess_image_for_ocr(pil_image):
    img = np.array(pil_image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = cv2.medianBlur(img_bin, 3)
    kernel = np.ones((2, 2), np.uint8)
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)
    return Image.fromarray(img_bin)

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

def list_pdfs(base_folder):
    pdfs = []
    for file in os.listdir(base_folder):
        if file.lower().endswith(".pdf"):
            caminho_pdf = os.path.join(base_folder, file)
            pdfs.append(caminho_pdf)
    return pdfs

def carregar_modelo():
    if os.path.isfile(MODEL_FILENAME):
        modelo = joblib.load(MODEL_FILENAME)
        print(f"✅ Modelo carregado de '{MODEL_FILENAME}'")
        return modelo
    else:
        print(f"❌ Arquivo '{MODEL_FILENAME}' não encontrado.")
        return None

def classificar_pdf(modelo, pdf_path):
    texto = extract_text_from_pdf(pdf_path)
    texto = preprocess_text(texto)
    if not texto:
        return "Erro ao extrair texto"
    return modelo.predict([texto])[0]

# --------- EXECUÇÃO ---------
modelo = carregar_modelo()
if modelo:
    print("\n🔍 Classificando PDFs na pasta:")
    pdfs = list_pdfs(FOLDER_PATH)

    for pdf in pdfs:
        if pdf.endswith(".pkl"):
            continue  # ignora o modelo

        print(f"\n⏱️ Iniciando processamento de: {os.path.basename(pdf)}")
        inicio = time.time()

        tipo = classificar_pdf(modelo, pdf)

        fim = time.time()
        duracao = fim - inicio

        print(f"📄 {os.path.basename(pdf)} → 🏷️ Tipo classificado: {tipo}")
        print(f"⏳ Tempo gasto: {duracao:.2f} segundos")