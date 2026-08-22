import os
import pickle
import faiss
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

REPO_ID = "ai4bharat/MSMARCO-XI"

SHARDS = [
    "tamtrain.parquet",
    "hintrain.parquet",
    "bentrain.parquet",
    "gujtrain.parquet",
    "urdtrain.parquet"
]

TARGET_PER_SHARD = 10000
BATCH_SIZE = 512

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "index")
os.makedirs(INDEX_DIR, exist_ok=True)

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

index = None
texts = []
metadata = []

for shard_name in SHARDS:
    print(f"\nProcessing {shard_name}")

    shard_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"train/{shard_name}"
    )

    pf = pq.ParquetFile(shard_path)
    language = shard_name[:3]
    collected = 0

    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        table = batch.to_pydict()

        remaining = TARGET_PER_SHARD - collected
        if remaining <= 0:
            break

        answers = table["Answer"][:remaining]
        queries = table["query"][:remaining]
        query_ids = table["query_id"][:remaining]

        embeddings = model.encode(
            answers,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        if index is None:
            index = faiss.IndexFlatL2(embeddings.shape[1])

        index.add(embeddings)
        texts.extend(answers)

        metadata.extend([
            {
                "query": q,
                "query_id": qid,
                "language": language,
                "source": "MSMARCO-XI",
                "chunk_type": "raw"
            }
            for q, qid in zip(queries, query_ids)
        ])

        collected += len(answers)
        del embeddings

    print(f"Finished {shard_name}: {collected}")

faiss.write_index(index, os.path.join(INDEX_DIR, "msmarco.index"))

with open(os.path.join(INDEX_DIR, "texts.pkl"), "wb") as f:
    pickle.dump(texts, f)

with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print("\nDone!")
print(f"Indexed {len(texts)} passages.")