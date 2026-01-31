from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router
from app.services.index_manager import initialize_index, start_auto_refresh


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 FastAPI startup initiated")

    # Initialize index ONLY (fast operation)
    try:
        initialize_index()
        print("✅ FAISS index initialized")
    except Exception as e:
        print(f"❌ Index init failed: {e}")

    # Start auto refresh in background (non-blocking)
    start_auto_refresh(900)

    yield  # 👈 app is now ready to accept requests

    print("🛑 FastAPI shutdown")


app = FastAPI(
    title="Job Recommendation API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
