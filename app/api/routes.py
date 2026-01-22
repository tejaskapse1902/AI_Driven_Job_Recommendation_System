from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from app.services.resume_parser import parse_resume_file
from app.services.recommender import recommend_jobs
from app.services.s3_service import upload_to_s3, list_resumes, delete_resume
import os
import tempfile

router = APIRouter()


class DeleteRequest(BaseModel):
    key: str
    

@router.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        s3_key = upload_to_s3(tmp_path, file.filename)
        resume_text = parse_resume_file(file)
        results = recommend_jobs(resume_text)


        return {
        "filename": file.filename,
        "recommendations": results
        }
    finally:
        os.remove(tmp_path)
        
@router.get("/resumes")
def get_all_resumes():
    return list_resumes()


@router.delete("/resumes")
def delete_resume_api(req: DeleteRequest):
    delete_resume(req.key)
    return {"status": "deleted"}