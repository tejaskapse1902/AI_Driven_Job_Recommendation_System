import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# -------- CONFIG --------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

JOBS_CSV = os.path.join(DATA_DIR, "jobs.csv")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "jobs.index")

MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32
# ------------------------


def build_job_text(row):
    return f"""
Job Title: {row.get('Job Title', '')}
Category: {row.get('Category', '')}
Experience Level: {row.get('Experience Level', '')}
Skills: {row.get('Skills', '')}
Requirements: {row.get('Requirements', '')}
Responsibilities: {row.get('Responsibilities', '')}
Job Description: {row.get('Job Description', '')}
"""


def main():
    print("📥 Loading jobs.csv...")
    df = pd.read_csv(JOBS_CSV, encoding="latin1")
    print(f"Jobs loaded: {len(df)}")

    print("🧱 Building job texts...")
    job_texts = df.apply(build_job_text, axis=1).tolist()

    print("🤖 Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("⚙️ Generating embeddings...")
    embeddings = model.encode(
        job_texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    dim = embeddings.shape[1]

    print("🧠 Creating FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))

    print("💾 Saving FAISS index...")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("✅ Done!")
    print("Index saved to:", FAISS_INDEX_PATH)


if __name__ == "__main__":
    main()
