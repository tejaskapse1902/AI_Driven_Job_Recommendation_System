import sys
import os
import dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

ENV_PATH = os.path.join(PROJECT_ROOT, "app", ".env")
dotenv.load_dotenv(ENV_PATH)

import boto3
from app.core.config import DATA_DIR

# ---------- CONFIG ----------
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
S3_KEY = "faiss/jobs.index"     # path inside bucket
LOCAL_INDEX_PATH = f"{DATA_DIR}/jobs.index"
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
# ---------------------------

def upload_index():
    if not os.path.exists(LOCAL_INDEX_PATH):
        raise FileNotFoundError("jobs.index not found")

    s3 = boto3.client("s3", region_name=AWS_REGION)

    s3.upload_file(
        LOCAL_INDEX_PATH,
        BUCKET_NAME,
        S3_KEY
    )

    print("✅ jobs.index uploaded to S3")

if __name__ == "__main__":
    upload_index()
