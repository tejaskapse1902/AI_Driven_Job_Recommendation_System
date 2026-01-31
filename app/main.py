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

import threading

@app.on_event("startup")
def startup_event():
    def init_task():
        try:
            print("🚀 Initializing recommendation system in background...")
            # get_model()  # Preload model
            initialize_index()
            print("✅ Recommendation system ready.")
        except Exception as e:
            print(f"❌ Background initialization failed: {e}")

    # Run initialization in a background thread to prevent startup timeout (502 errors)
    threading.Thread(target=init_task, daemon=True).start()
    
    start_auto_refresh(900)   # 15 minutes
    print("🚀 API is starting up... (Background tasks running)")
