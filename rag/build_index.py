"""
Tiny RAG index builder stub.
This script demonstrates how to produce a small JSON 'index' suitable for tests or demos.
Replace with real embeddings + vector store logic (e.g. FAISS, Chroma).
"""
import json
from pathlib import Path

def build_sample_index(output: str = "aegis_multimodal_ai_system/rag/sample_index.json"):
    data = [
        {"id": "doc1", "text": "This is a sample document about cats."},
        {"id": "doc2", "text": "This document discusses safety and policy."},
    ]
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    print(f"Wrote sample index to {output}")

if __name__ == "__main__":
    build_sample_index()
