"""Research-only image classification API. Never present results as a diagnosis."""

import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from model import CLASSES, SkinLesionCNN, VALID_TRANSFORM

CHECKPOINT_PATH = Path(os.getenv("MODEL_CHECKPOINT", "checkpoints/best.pt"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
app = FastAPI(title="Skin lesion research classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:5173").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
model = None


def get_model():
    global model
    if model is not None:
        return model
    if not CHECKPOINT_PATH.is_file():
        return None
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        if checkpoint.get("classes") != list(CLASSES):
            raise ValueError("Checkpoint classes do not match this API")
        loaded = SkinLesionCNN()
        loaded.load_state_dict(checkpoint["model_state"])
        loaded.eval()
        model = loaded
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail="Checkpoint could not be loaded; retrain the model") from exc
    return model


@app.get("/api/health")
def health():
    return {"status": "ok", "model_ready": CHECKPOINT_PATH.is_file(), "classes": list(CLASSES)}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    classifier = get_model()
    if classifier is None:
        raise HTTPException(status_code=503, detail="No trained model available. Train on HAM10000 first.")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            tensor = VALID_TRANSFORM(image.convert("RGB")).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    with torch.inference_mode():
        probabilities = torch.softmax(classifier(tensor), dim=1)[0].tolist()
    scores = sorted(
        ({"code": code, "probability": round(score, 6)} for code, score in zip(CLASSES, probabilities)),
        key=lambda item: item["probability"],
        reverse=True,
    )
    return {"predicted_class": scores[0]["code"], "scores": scores, "disclaimer": "Research use only. Not a medical diagnosis."}
