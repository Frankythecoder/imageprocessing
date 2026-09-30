import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as api
from model import CLASSES, SkinLesionCNN
from train import split_by_lesion


def test_model_output_shape():
    import torch

    network = SkinLesionCNN().eval()
    with torch.inference_mode():
        assert network(torch.zeros(1, 3, 64, 64)).shape == (1, len(CLASSES))


def test_health_and_missing_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "CHECKPOINT_PATH", tmp_path / "absent.pt")
    monkeypatch.setattr(api, "model", None)
    client = TestClient(api.app)
    assert client.get("/api/health").json()["model_ready"] is False
    response = client.post("/api/predict", files={"file": ("test.png", b"not an image", "image/png")})
    assert response.status_code == 503


def test_lesion_split_prevents_leakage():
    rows = []
    for label in CLASSES:
        for index in range(4):
            for photo in range(2):
                rows.append({"lesion_id": f"{label}-{index}", "image_id": f"{label}-{index}-{photo}", "dx": label})
    train, validation = split_by_lesion(pd.DataFrame(rows), 0.25, 42)
    assert set(train.lesion_id).isdisjoint(validation.lesion_id)
    assert set(train.dx) == set(validation.dx) == set(CLASSES)
