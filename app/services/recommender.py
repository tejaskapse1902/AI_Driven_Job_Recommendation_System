# =============================
# app/services/recommender.py
# Accuracy-safe + fast version
# Uses TOP_K = 20 (candidate set)
# Works with IndexHNSWFlat built in index builder
# =============================

import os
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"

import numpy as np
from sentence_transformers import SentenceTransformer
from app.services.resume_parser import parse_resume
from app.services.index_manager import get_index, get_jobs_df
from datetime import datetime, timezone
import re

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 20   # candidate pool size (fast + no accuracy loss)

# ---------- Load model once (singleton) ----------
_model = None


def get_model():
    global _model
    if _model is None:
        print("🔥 Loading embedding model once...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ---------- Utils ----------

def clean_job_link(raw):
    if not raw:
        return ""

    raw = raw.strip()

    if "@" in raw and "http" not in raw:
        return f"mailto:{raw}"

    raw = raw.replace("https: ", "https://")
    raw = raw.replace("http: ", "http://")
    raw = raw.replace(" ", "")

    match = re.search(r"(https?://[^\s]+)", raw)
    if match:
        return match.group(1)

    return raw



def recency_boost(created_date, max_boost=0.08, decay_days=30):
    try:
        if isinstance(created_date, str):
            created_dt = datetime.fromisoformat(created_date.replace("Z", "+00:00"))
        elif isinstance(created_date, datetime):
            created_dt = created_date
        else:
            return 0.0

        now = datetime.now(timezone.utc)
        age_days = max((now - created_dt).days, 0)

        boost = max_boost * max(0, (decay_days - age_days) / decay_days)
        return boost

    except Exception:
        return 0.0



def final_score(similarity, row, resume_data):
    score = similarity

    job_skills = str(row.get("Skills", "")).lower()
    overlap = sum(1 for s in resume_data["skills"] if s in job_skills)
    score += 0.07 * overlap

    if resume_data.get("experience_years"):
        if str(resume_data["experience_years"]) in str(row.get("Experience Level", "")):
            score += 0.15

    score += recency_boost(row.get("created_date"))

    return score


# ---------- Main recommender ----------

def recommend_jobs(resume_text: str):
    index = get_index()
    df = get_jobs_df()

    if index is None or df is None or df.empty:
        return {
            "error": "Recommendation system is warming up. Please try again in a few seconds."
        }

    resume_data = parse_resume(resume_text)

    model = get_model()

    # Keep batch-style encoding for identical accuracy
    emb_vec = model.encode([resume_text], normalize_embeddings=True)[0]
    emb = np.asarray([emb_vec], dtype="float32")

    scores, indices = index.search(emb, TOP_K)

    ranked = []
    for rank, idx in enumerate(indices[0]):
        if idx >= len(df):
            continue

        row = df.iloc[idx]
        sim = float(scores[0][rank])
        score = final_score(sim, row, resume_data)

        created_date = row.get("created_date")
        try:
            if isinstance(created_date, str):
                created_dt = datetime.fromisoformat(created_date.replace("Z", "+00:00"))
            else:
                created_dt = created_date
        except Exception:
            created_dt = datetime.min

        ranked.append((score, created_dt, idx))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)

    results = []
    for score, _, idx in ranked[:20]:
        job = df.iloc[idx]

        results.append({
            "job_title": str(job.get("Job Title", "")),
            "company": str(job.get("Company Name", "")),
            "location": str(job.get("Location", "")),
            "experience": str(job.get("Experience Level", "")),
            "skills": str(job.get("Skills", "")),
            "salary_min": str(job.get("Salary Min (?)")),
            "salary_max": str(job.get("Salary Max (?)")),
            "match_percentage": round(min(score * 100, 100), 2),
            "created_date": str(job.get("created_date", "")),
            "job_link": clean_job_link(job.get("Direct Link", "")),
        })

    return results
