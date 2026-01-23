import numpy as np
import pandas as pd
import faiss
from app.core.config import DATA_DIR
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ===== CONFIG =====
MONGO_URI = "mongodb+srv://tejaskapse19_db_user:BEsS1fFSuSLWZLCM@cluster0.yknizjc.mongodb.net/?appName=Cluster0"
DB_NAME = "job_recommendation"
COLLECTION_NAME = "jobs"
OUTPUT_INDEX = f"{DATA_DIR}/jobs.index"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
# ==================

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
    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    jobs = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(jobs)

    print("Building job texts...")
    job_texts = df.apply(build_job_text, axis=1).tolist()

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(job_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)

    print("Creating FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))

    print("Saving index...")
    faiss.write_index(index, OUTPUT_INDEX)

    print("Index created successfully.")

if __name__ == "__main__":
    main()
