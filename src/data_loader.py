import random
import shutil
from pathlib import Path
import cv2
import numpy as np
from src.config import CLASSIFICATION_CLASSES, PROCESSED_DATA_DIR, RAW_DATA_DIR, YOLO_DATA_DIR

def validate_image_integrity(path):
    try:
        img = cv2.imread(str(path))
        return img is not None and img.size > 0
    except:
        return False

def augment_image(image):
    aug = image.copy()
    flip = random.choice([0, 1, -1, None])
    if flip is not None:
        aug = cv2.flip(aug, flip)
    if random.random() < 0.3:
        aug = cv2.GaussianBlur(aug, (5, 5), 0)
    return aug

def split_classification_data(raw_dir=None, dest_dir=None):
    raw = Path(raw_dir or RAW_DATA_DIR)
    dest = Path(dest_dir or (PROCESSED_DATA_DIR / "classification"))
    for cls in CLASSIFICATION_CLASSES:
        files = [p for p in (raw / cls).glob("*.*") if validate_image_integrity(p)]
        random.shuffle(files)
        n = len(files)
        splits = {
            "train": files[:int(n * 0.7)],
            "val": files[int(n * 0.7):int(n * 0.85)],
            "test": files[int(n * 0.85):],
        }
        for split, split_files in splits.items():
            out = dest / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out / f.name)

def validate_yolo_labels(yolo_dir=None, max_samples=None):
    base = Path(yolo_dir or YOLO_DATA_DIR)
    stats = {}
    for split in ["train", "val"]:
        imgs = list((base / "images" / split).glob("*.jpg"))
        if max_samples:
            imgs = imgs[:max_samples]
        valid = sum(1 for img in imgs if (base / "labels" / split / f"{img.stem}.txt").exists())
        stats[split] = {"total_images": len(imgs), "valid_pairs": valid}
    return stats

def load_dataset_arrays(folder, size=(224, 224)):
    X, y = [], []
    for idx, cls in enumerate(CLASSIFICATION_CLASSES):
        p = Path(folder) / cls
        if p.exists():
            for f in p.glob("*.*"):
                img = cv2.imread(str(f))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    X.append(cv2.resize(img, size))
                    y.append(idx)
    return np.array(X, dtype=np.float32) / 255.0, np.array(y, dtype=np.int32)
