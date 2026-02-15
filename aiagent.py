import subprocess
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Reload FAISS index
index = faiss.read_index(r"C:\Users\AMIT K\OneDrive\Desktop\PYTHON\LEARNING\Personal agent\ppt_index.faiss")

# Load text lines
with open("ppt_text_output.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ask a question
query = input('Ask your query:')
query_embedding = model.encode([query])
D, I = index.search(np.array(query_embedding), k=3)

# Collect retrieved context
retrieved_texts = [lines[idx] for idx in I[0]]
context = "\n".join(retrieved_texts)

# Call Ollama locally (using mistral model)
prompt = f"Answer the question based on this context:\n{context}\n\nQuestion: {query}"
result = subprocess.run(
    ["ollama", "run", "mistral"],
    input=prompt.encode("utf-8"),
    capture_output=True
)

print("Answer:", result.stdout.decode("utf-8"))