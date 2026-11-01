from pathlib import Path
import numpy as np
import pytest

from src.config import RAW_DATA_DIR
from src.data_loader import (
    augment_image,
    validate_image_integrity,
    validate_yolo_labels,
)


def get_first_sample_image() -> Path:
    for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
        folder = RAW_DATA_DIR / cls
        if folder.exists():
            for p in folder.glob("*.jpg"):
                return p
    pytest.skip("No sample image found in data/raw")


def test_validate_image_integrity_valid():
    img_path = get_first_sample_image()
    assert validate_image_integrity(img_path) is True


def test_validate_image_integrity_invalid():
    fake_path = Path("non_existent_image.xyz")
    assert validate_image_integrity(fake_path) is False


def test_augment_image():
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    aug = augment_image(dummy_img)
    assert aug.shape == dummy_img.shape
    assert aug.dtype == np.uint8


def test_validate_yolo_labels():
    stats = validate_yolo_labels(max_samples=50)
    assert isinstance(stats, dict)
    for split in ["train", "val"]:
        if split in stats:
            assert "total_images" in stats[split]
            assert "valid_pairs" in stats[split]
