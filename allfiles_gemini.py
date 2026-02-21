import os
import numpy as np
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from pptx import Presentation
from docx import Document
import pdfplumber
import google.generativeai as genai
from sklearn.neighbors import NearestNeighbors   # ⭐ FAISS-free retrieval

app = Flask(__name__)

# -----------------------------
# Lazy Globals
# -----------------------------
model = None
nn_model = None
all_texts = None
embeddings = None
genai_configured = False


# -----------------------------
# Extract text from PPT
# -----------------------------
def extract_text_from_ppt(filepath):
    texts = []
    prs = Presentation(filepath)
    for i, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    if para.text.strip():
                        texts.append(f"{os.path.basename(filepath)} - Slide {i}: {para.text.strip()}")
    return texts


# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_text_from_pdf(filepath):
    texts = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                texts.append(f"{os.path.basename(filepath)} - Page {i}: {text.strip()}")
    return texts


# -----------------------------
# Ingest folder (lazy)
# -----------------------------
def ingest_folder(folder_path):
    texts = []
    if not os.path.exists(folder_path):
        return texts

    for filename in os.listdir(folder_path):
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


# -----------------------------
# Highlight helper
# -----------------------------
def highlight_terms(text, query):
    for term in query.split():
        text = text.replace(term, f"<mark>{term}</mark>")
        text = text.replace(term.capitalize(), f"<mark>{term.capitalize()}</mark>")
    return text


# -----------------------------
# Home Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    global model, nn_model, all_texts, embeddings, genai_configured

    answer = ""
    context = ""

    if request.method == "POST":
        query = request.form["query"]

        # Load documents
        if all_texts is None:
            all_texts = ingest_folder("ALL_Docs")

        if not all_texts:
            return "No documents found in ALL_Docs folder."

        # Load embedding model
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")

        # Build embeddings + NearestNeighbors
        if embeddings is None:
            embeddings = model.encode(all_texts)
            nn_model = NearestNeighbors(n_neighbors=5, metric="cosine")
            nn_model.fit(embeddings)

        # Configure Gemini once
        if not genai_configured:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            genai_configured = True

        # Search
        query_embedding = model.encode([query])
        distances, indices = nn_model.kneighbors(query_embedding)

        context = "\n".join([highlight_terms(all_texts[idx], query) for idx in indices[0]])

        # Gemini response
        response = genai.generate_text(
            model="models/gemini-2.5-flash",
            prompt=(
                f"Answer the question strictly using the context below.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Give a direct answer based only on the context."
            )
        )

        answer = response.result

    return render_template("index.html", answer=answer, context=context)


# -----------------------------
# Run locally
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
