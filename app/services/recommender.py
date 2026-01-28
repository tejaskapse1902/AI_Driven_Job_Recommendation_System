from operator import index
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from app.services.resume_parser import parse_resume
from app.services.index_manager import get_index, get_jobs_df
from datetime import datetime, timezone
import re

# ---------- Load model once ----------
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

TOP_K = 50

def clean_job_link(raw):
    if not raw:
        return ""

    raw = raw.strip()

    # If email → convert to mailto
    if "@" in raw and "http" not in raw:
        return f"mailto:{raw}"

    # Fix broken https
    raw = raw.replace("https: ", "https://")
    raw = raw.replace("http: ", "http://")

    # Remove spaces inside URL
    raw = raw.replace(" ", "")

    # Extract first valid URL if mixed text
    match = re.search(r"(https?://[^\s]+)", raw)
    if match:
        return match.group(1)

    return raw



def recency_boost(created_date, max_boost=0.08, decay_days=30):
    """
    Boost score for newer jobs.
    max_boost: maximum bonus added to similarity score
    decay_days: after how many days boost becomes 0
    """
    try:
        if isinstance(created_date, str):
            created_dt = datetime.fromisoformat(created_date.replace("Z", "+00:00"))
        elif isinstance(created_date, datetime):
            created_dt = created_date
        else:
            return 0.0

        now = datetime.now(timezone.utc)
        age_days = (now - created_dt).days
        if age_days < 0:
            age_days = 0

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

    # 🔥 Date / recency boost
    created_date = row.get("created_date")
    score += recency_boost(created_date)

    return score


def recommend_jobs(resume_text: str):
    # ✅ Always fetch latest index + jobs
    index = get_index()
    df = get_jobs_df()

    if index is None or df is None or df.empty:
        return {
            "error": "Recommendation system is warming up. Please try again in a few seconds."
        }

    resume_data = parse_resume(resume_text)

    emb = model.encode([resume_text], normalize_embeddings=True)
    scores, indices = index.search(np.array(emb), TOP_K)


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
        except:
            created_dt = datetime.min

        ranked.append((score, created_dt, idx))

    # ✅ Sort: highest score first, then newest job
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)

    results = []
    for score, _, idx in ranked[:TOP_K]:
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
