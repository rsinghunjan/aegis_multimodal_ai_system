from fastapi import FastAPI

from .multimodal_ai_system import MultimodalAISystem

app = FastAPI(title="Aegis Multimodal AI System")
ai_system = MultimodalAISystem()

@app.get("/")
def root():
    return {"status": "Aegis Multimodal AI System is running."}

# Add more endpoints as needed, e.g., /chat, /image, etc.