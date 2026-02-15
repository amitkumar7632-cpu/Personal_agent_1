from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load your text file
with open("ppt_text_output.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Step 1: Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Convert text lines into embeddings
embeddings = model.encode(lines)

# Step 3: Store embeddings in FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("Embeddings stored in FAISS index!")

# Step 4: Example query
query = input("Ask your question here:")
query_embedding = model.encode([query])
D, I = index.search(np.array(query_embedding), k=3)

print("\nTop matches:")
for idx in I[0]:
    print(lines[idx])