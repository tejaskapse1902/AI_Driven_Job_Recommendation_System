import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "job_recommendation"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

jobs_collection = db["jobs"]
