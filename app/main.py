from fastapi import FastAPI
from app.api.routes import router
from app.core.startup import start_background_loading
import app.core.state as state
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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
    # 🔥 non-blocking startup
    start_background_loading()


@app.get("/")
def health():
    return {
        "status": "ok",
        "ready": state.READY
    }
