# Aegis Multimodal AI System

## Structure

- `multimodal_ai_system.py` — Core multimodal AI logic.
- `safety_checker.py` — Safety checking utilities.
- `error_handling.py` — Custom error classes.
- `frontend.py` — Gradio UI demo.
- `app.py` — FastAPI API.
- `main.py` — Uvicorn entry point.

## Running the API

```bash
uvicorn aegis_multimodal_ai_system.main:app --reload
```

## Running the Frontend

```bash
python -m aegis_multimodal_ai_system.frontend
```
