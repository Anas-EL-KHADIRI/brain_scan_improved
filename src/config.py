import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
YOLO_DATA_DIR = DATA_DIR / "yolo"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CLASSIFICATION_MODEL_PATH = MODELS_DIR / "googlenet_brain_tumor.keras"
DETECTION_MODEL_PATH = MODELS_DIR / "yolov8_brain_tumor.pt"

CLASSIFICATION_IMG_SIZE = (224, 224)
DETECTION_IMG_SIZE = 640

CLASSIFICATION_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
DETECTION_CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

CLASS_DISPLAY_NAMES = {
    "glioma": "Gliome (Glioma)",
    "meningioma": "Méningiome (Meningioma)",
    "notumor": "Sain / Pas de tumeur (No Tumor)",
    "no_tumor": "Sain / Pas de tumeur (No Tumor)",
    "pituitary": "Tumeur pituitaire (Pituitary)",
}

CLASS_ALERT_LEVEL = {
    "glioma": "high",
    "meningioma": "medium",
    "pituitary": "medium",
    "notumor": "normal",
    "no_tumor": "normal",
}

MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
CLASSIFICATION_EXPERIMENT_NAME = "Brain_Tumor_Classification"
DETECTION_EXPERIMENT_NAME = "Brain_Tumor_Detection"

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = f"http://{API_HOST}:{API_PORT}"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
MAX_IMAGE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
