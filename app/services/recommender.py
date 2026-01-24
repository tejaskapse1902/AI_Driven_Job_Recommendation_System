import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from app.services.resume_parser import parse_resume
from app.services.index_manager import get_index, get_jobs_df

# ---------- Load model once ----------
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

TOP_K = 50


def final_score(similarity, row, resume_data):
    score = similarity

    job_skills = str(row.get("Skills", "")).lower()
    overlap = sum(1 for s in resume_data["skills"] if s in job_skills)
    score += 0.07 * overlap

    if resume_data.get("experience_years"):
        if str(resume_data["experience_years"]) in str(row.get("Experience Level", "")):
            score += 0.15

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
        if idx >= len(df):   # safety check (important for incremental index)
            continue

        row = df.iloc[idx]
        sim = float(scores[0][rank])
        score = final_score(sim, row, resume_data)
        ranked.append((score, idx))

    ranked.sort(reverse=True)

    results = []
    for score, idx in ranked[:TOP_K]:
        job = df.iloc[idx]

        results.append({
            "job_title": str(job.get("Job Title", "")),
            "company": str(job.get("Company Name", "")),
            "location": str(job.get("Location", "")),
            "experience": str(job.get("Experience Level", "")),
            "skills": str(job.get("Skills", "")),
            "salary_min": str(job.get("Salary Min (?)")),
            "salary_max": str(job.get("Salary Max (?)")),
            "match_percentage": round(min(score * 100, 100), 2)
        })

    return results
