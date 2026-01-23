import os
import numpy as np
import faiss
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import boto3
import dotenv
from app.core.config import DATA_DIR

dotenv.load_dotenv()

# ---------------- CONFIG ----------------
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "job_recommendation"
COLLECTION = "jobs"

BUCKET = os.getenv("AWS_BUCKET_NAME")
S3_KEY = "faiss/jobs.index"
LOCAL_INDEX = f"{DATA_DIR}/jobs.index"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
# ----------------------------------------


def build_job_text(job):
    return f"""
Job Title: {job.get('Job Title', '')}
Category: {job.get('Category', '')}
Experience Level: {job.get('Experience Level', '')}
Skills: {job.get('Skills', '')}
Requirements: {job.get('Requirements', '')}
Responsibilities: {job.get('Responsibilities', '')}
Job Description: {job.get('Job Description', '')}
"""


def download_existing_index():
    s3 = boto3.client("s3")
    try:
        s3.download_file(BUCKET, S3_KEY, LOCAL_INDEX)
        print("Existing index downloaded")
        return faiss.read_index(LOCAL_INDEX)
    except:
        print("No existing index found, creating new")
        return None


def upload_index():
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    s3.upload_file(LOCAL_INDEX, BUCKET, S3_KEY)
    print("Updated index uploaded to S3")


def main():
    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]

    new_jobs = list(col.find({"indexed": {"$ne": True}}))

    if not new_jobs:
        print("No new jobs to index")
        return

    print(f"New jobs found: {len(new_jobs)}")

    job_texts = [build_job_text(j) for j in new_jobs]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(job_texts, batch_size=32, normalize_embeddings=True)

    index = download_existing_index()

    if index is None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)

    index.add(np.array(embeddings))

    faiss.write_index(index, LOCAL_INDEX)

    upload_index()

    # mark jobs indexed
    ids = [j["_id"] for j in new_jobs]
    col.update_many({"_id": {"$in": ids}}, {"$set": {"indexed": True}})

    print("Index updated successfully")


if __name__ == "__main__":
    main()
