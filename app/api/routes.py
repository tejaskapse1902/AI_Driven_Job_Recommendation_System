from fastapi import APIRouter, UploadFile, File
from app.services.resume_parser import parse_resume_file
from app.services.recommender import recommend_jobs

router = APIRouter()

@router.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    resume_text = parse_resume_file(file)
    results = recommend_jobs(resume_text)


    return {
    "filename": file.filename,
    "recommendations": results
    }