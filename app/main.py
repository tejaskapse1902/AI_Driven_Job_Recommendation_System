from fastapi import FastAPI
from app.api.routes import router
from app.services.index_manager import initialize_index, start_auto_refresh
from fastapi.middleware.cors import CORSMiddleware

from app.services.recommender import get_model

app = FastAPI(title="Job Recommendation API")

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)


app.include_router(router)

@app.on_event("startup")
def startup_event():
    print("🚀 Starting up Job Recommendation API...")
    get_model()  # Preload model at startup
    initialize_index()
    start_auto_refresh(900)   # 15 minutes
    print("✅ Startup complete.")