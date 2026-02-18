import os
import numpy as np
import faiss
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from pptx import Presentation
from docx import Document
import pdfplumber
from google import genai

app = Flask(__name__)

# -----------------------------
# Lazy Globals (loaded on demand)
# -----------------------------
model = None
index = None
all_texts = None
client = None


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
    global model, index, all_texts, client

    answer = ""
    context = ""

    # Lazy load everything only when needed
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

        # Build FAISS index
        if index is None:
            embeddings = model.encode(all_texts)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))

        # Load Gemini client
        if client is None:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Search
        query_embedding = model.encode([query])
        D, I = index.search(np.array(query_embedding), k=5)

        context = "\n".join([highlight_terms(all_texts[idx], query) for idx in I[0]])

        # Gemini response
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

    return render_template("index.html", answer=answer, context=context)


# -----------------------------
# Run locally
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
