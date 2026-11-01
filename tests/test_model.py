from pathlib import Path
import pytest
from src.config import (
    CLASSIFICATION_CLASSES,
    CLASSIFICATION_MODEL_PATH,
    DETECTION_MODEL_PATH,
    RAW_DATA_DIR,
)
from src.predict import BrainClassifier, BrainTumorDetector


def get_first_sample_image() -> Path:
    for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
        folder = RAW_DATA_DIR / cls
        if folder.exists():
            for p in folder.glob("*.jpg"):
                return p
    pytest.skip("No sample image found")


def test_model_files_exist():
    assert CLASSIFICATION_MODEL_PATH.exists(), f"Missing {CLASSIFICATION_MODEL_PATH}"
    assert DETECTION_MODEL_PATH.exists(), f"Missing {DETECTION_MODEL_PATH}"


def test_classifier_inference():
    sample_path = get_first_sample_image()
    classifier = BrainClassifier()
    result = classifier.predict(sample_path)

    assert "predicted_class" in result
    assert result["predicted_class"] in CLASSIFICATION_CLASSES
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 100
    assert "probabilities" in result
    assert len(result["probabilities"]) == len(CLASSIFICATION_CLASSES)
    assert result["latency_ms"] > 0


def test_detector_inference():
    sample_path = get_first_sample_image()
    detector = BrainTumorDetector()
    result = detector.predict(sample_path, conf_threshold=0.25)

    assert "num_detections" in result
    assert isinstance(result["num_detections"], int)
    assert "detections" in result
    assert isinstance(result["detections"], list)
    assert "annotated_image_base64" in result
    assert result["latency_ms"] > 0
