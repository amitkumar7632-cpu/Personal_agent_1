import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load text lines from your slides
with open(r"C:\Users\AMIT K\OneDrive\Desktop\PYTHON\LEARNING\Personal agent\ppt_text_output.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(lines)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index to disk
faiss.write_index(index, r"C:\Users\AMIT K\OneDrive\Desktop\PYTHON\LEARNING\Personal agent\ppt_index.faiss")
print("Index saved successfully!")