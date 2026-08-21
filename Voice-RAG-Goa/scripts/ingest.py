from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from pathlib import Path

# Root project directory
ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"
INDEX_DIR.mkdir(exist_ok=True)

print("Loading MSMARCO-XI dataset (streaming mode)...")

dataset = load_dataset(
    "ai4bharat/MSMARCO-XI",
    split="train",
    streaming=True
)

texts = []

# Collect only 10,000 passages
for item in dataset:
    if "passage" in item:
        texts.append(item["passage"])
    elif "text" in item:
        texts.append(item["text"])

    if len(texts) >= 10000:
        break

print(f"Collected {len(texts)} passages.")

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Creating embeddings...")
embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, str(INDEX_DIR / "msmarco.index"))

with open(INDEX_DIR / "texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("Done!")
print(f"Saved {len(texts)} passages.")