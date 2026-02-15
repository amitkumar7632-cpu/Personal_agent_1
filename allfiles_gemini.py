import os
import numpy as np
import faiss
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from pptx import Presentation
from docx import Document
import pdfplumber
from PIL import Image
import pytesseract
from google import genai

# --- Explicitly set Tesseract path ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Render!"

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Render provides PORT automatically
    app.run(host="0.0.0.0", port=port)
    
# --- Extract text from PPT slides (shapes + images with OCR) ---
def extract_text_from_ppt(filepath):
    texts = []
    prs = Presentation(filepath)
    for i, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            # Text frames
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    if para.text.strip():
                        texts.append(f"{os.path.basename(filepath)} - Slide {i}: {para.text.strip()}")

            # Images (pictures)
            if shape.shape_type == 13:  # 13 = Picture
                image = shape.image
                image_bytes = image.blob
                temp_path = f"temp_slide_{i}.png"
                with open(temp_path, "wb") as f:
                    f.write(image_bytes)
                try:
                    img = Image.open(temp_path)
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        texts.append(f"{os.path.basename(filepath)} - Slide {i} (OCR): {ocr_text.strip()}")
                except Exception as e:
                    print("OCR error:", e)
                finally:
                    os.remove(temp_path)
    return texts

# --- Extract text from PDF (text + OCR fallback) ---
def extract_text_from_pdf(filepath):
    texts = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                texts.append(f"{os.path.basename(filepath)} - Page {i}: {text.strip()}")
            else:
                # Fallback: OCR on scanned pages
                img = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    texts.append(f"{os.path.basename(filepath)} - Page {i} (OCR): {ocr_text.strip()}")
    return texts

# --- Ingest all supported files from a folder ---
def ingest_folder(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        # Skip hidden/lock files created by Office
        if filename.startswith("~$"):
            continue

        filepath = os.path.join(folder_path, filename)

        if filename.lower().endswith(".pptx"):
            texts.extend(extract_text_from_ppt(filepath))

        elif filename.lower().endswith(".pdf"):
            texts.extend(extract_text_from_pdf(filepath))

        elif filename.lower().endswith(".docx"):
            doc = Document(filepath)
            for i, para in enumerate(doc.paragraphs, start=1):
                if para.text.strip():
                    texts.append(f"{filename} - Paragraph {i}: {para.text.strip()}")

        elif filename.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f.readlines(), start=1):
                    if line.strip():
                        texts.append(f"{filename} - Line {i}: {line.strip()}")

    return texts

# --- Load all documents from your folder ---
folder_path = r"C:\Users\AMIT K\OneDrive\ALL_Docs"
all_texts = ingest_folder(folder_path)

# --- Embeddings + FAISS index ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(all_texts)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# --- Gemini client (direct API key) ---
client = genai.Client(api_key="AIzaSyAqXiJcrigh5bxeiWPnspnLcOWyCRcGqVE")

# --- Flask app ---
app = Flask(__name__)

# --- Helper function for highlighting query terms ---
def highlight_terms(text, query):
    # Case-insensitive highlighting
    for term in query.split():
        # Replace lowercase and capitalized versions
        text = text.replace(term, f"<mark>{term}</mark>")
        text = text.replace(term.capitalize(), f"<mark>{term.capitalize()}</mark>")
    return text


@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    context = ""
    if request.method == "POST":
        query = request.form["query"]
        query_embedding = model.encode([query])
        D, I = index.search(np.array(query_embedding), k=5)

        # Highlight query terms in retrieved context
        context = "\n".join([highlight_terms(all_texts[idx], query) for idx in I[0]])

        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=(
                f"Answer the question strictly using the context below.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Give a direct answer based only on the context."
            )
        )
        answer = response.text

    return render_template("download_index.html", answer=answer, context=context)

if __name__ == "__main__":
    app.run(debug=True)