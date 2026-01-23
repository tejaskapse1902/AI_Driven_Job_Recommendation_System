import boto3
import os
import dotenv
from app.core.config import DATA_DIR

dotenv.load_dotenv()

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
