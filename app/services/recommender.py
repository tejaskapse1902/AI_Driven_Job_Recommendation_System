import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from app.core.config import DATA_DIR
from app.services.resume_parser import parse_resume

# Load data
df = pd.read_csv(f"{DATA_DIR}/jobs.csv", encoding="latin1")
df["Job Description"] = df["Job Description"].fillna("").astype(str)

index = faiss.read_index(f"{DATA_DIR}/jobs.index")

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
    resume_data = parse_resume(resume_text)

    emb = model.encode([resume_text], normalize_embeddings=True)
    scores, indices = index.search(np.array(emb), TOP_K)


    ranked = []
    for rank, idx in enumerate(indices[0]):
        row = df.iloc[idx]
        sim = scores[0][rank]
        score = final_score(sim, row, resume_data)
        ranked.append((score, idx))

    ranked.sort(reverse=True)

    results = []
    for score, idx in ranked:
        job = df.iloc[idx]
        results.append({
            "job_title": str(job.get("Job Title", "")),
            "company": str(job.get("Company Name", "")),
            "location": str(job.get("Location", "")),
            "experience": str(job.get("Experience Level", "")),
            "skills": str(job.get("Skills", "")),
            "salary_min": job.get("Salary Min (?)", 0) if pd.notna(job.get("Salary Min (?)")) else 0,
            "salary_max": job.get("Salary Max (?)", 0) if pd.notna(job.get("Salary Max (?)")) else 0,
            "match_percentage": float(round(min(float(score) * 100, 100), 2))
        })

    return results