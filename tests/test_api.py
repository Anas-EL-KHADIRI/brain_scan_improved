from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import RAW_DATA_DIR

client = TestClient(app)


def get_sample_image_path() -> Path:
    for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
        folder = RAW_DATA_DIR / cls
        if folder.exists():
            for f in folder.glob("*.jpg"):
                return f
    raise FileNotFoundError("No test image found in data/raw")


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "BrainScan AI API"
    assert data["status"] == "online"


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "models_loaded" in data


def test_classify_invalid_extension():
    fake_file = ("test.txt", b"not an image", "text/plain")
    response = client.post("/api/v1/classify", files={"file": fake_file})
    assert response.status_code == 400
    assert "Invalid file extension" in response.json()["detail"]


def test_classify_valid_image():
    sample_path = get_sample_image_path()
    with open(sample_path, "rb") as f:
        file_payload = {"file": (sample_path.name, f, "image/jpeg")}
        response = client.post("/api/v1/classify", files=file_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "predicted_class" in data
    assert "confidence_pct" in data
    assert 0 <= data["confidence_pct"] <= 100
    assert "probabilities" in data
    assert len(data["probabilities"]) == 4


def test_detect_valid_image():
    sample_path = get_sample_image_path()
    with open(sample_path, "rb") as f:
        file_payload = {"file": (sample_path.name, f, "image/jpeg")}
        response = client.post("/api/v1/detect?conf=0.20", files=file_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "num_detections" in data
    assert isinstance(data["detections"], list)
    assert data["annotated_image_base64"] is not None


def test_analyze_valid_image():
    sample_path = get_sample_image_path()
    with open(sample_path, "rb") as f:
        file_payload = {"file": (sample_path.name, f, "image/jpeg")}
        response = client.post("/api/v1/analyze?conf=0.20", files=file_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "classification" in data
    assert "detection" in data
    assert "total_latency_ms" in data
