import threading
from sentence_transformers import SentenceTransformer
from app.services import index_manager
import app.core.state as state

MODEL_NAME = "BAAI/bge-small-en-v1.5"

model = None
def get_model():
    global model
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    return model

def _load_resources():
    global model, READY

    try:
        print("🔥 Background loading started...")

        # 1️⃣ Load embedding model
        if model is None:
            model = SentenceTransformer(MODEL_NAME)
        print("✅ Embedding model loaded")

        # 2️⃣ Load FAISS + jobs (your existing logic)
        index_manager.initialize_index()

        # 3️⃣ Start auto refresh (already implemented by you)
        index_manager.start_auto_refresh(interval=900)

        state.READY = True
        print("🚀 System is READY")

    except Exception as e:
        print("❌ Startup failed:", e)


def start_background_loading():
    t = threading.Thread(target=_load_resources, daemon=True)
    t.start()
