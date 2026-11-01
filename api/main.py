from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from src.predict import BrainClassifier, BrainTumorDetector

app = FastAPI(title="BrainScan AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = BrainClassifier()
detector = BrainTumorDetector()

@app.get("/")
def root():
    return {
        "service": "BrainScan AI API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "health": "/api/v1/health",
            "classify": "/api/v1/classify",
            "detect": "/api/v1/detect",
            "analyze": "/api/v1/analyze",
        },
    }

@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "classifier": classifier.model is not None,
            "detector": detector.model is not None,
        },
        "api_version": "1.0.0",
    }

@app.post("/api/v1/classify")
async def classify(file: UploadFile = File(...)):
    name = file.filename or ""
    ext = ("." + name.split(".")[-1].lower()) if "." in name else ""
    if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
        raise HTTPException(status_code=400, detail=f"Invalid file extension '{ext}'.")
    content = await file.read()
    pred = classifier.predict(content)
    return {
        "status": "success",
        "predicted_class": pred["predicted_class"],
        "display_name": pred["display_name"],
        "confidence_pct": pred["confidence"],
        "probabilities": pred["probabilities"],
        "latency_ms": pred["latency_ms"],
        "image_shape": pred["image_shape"],
    }

@app.post("/api/v1/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.25):
    name = file.filename or ""
    ext = ("." + name.split(".")[-1].lower()) if "." in name else ""
    if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
        raise HTTPException(status_code=400, detail=f"Invalid file extension '{ext}'.")
    content = await file.read()
    pred = detector.predict(content, conf_threshold=conf, include_base64_image=True)
    return {
        "status": "success",
        "num_detections": pred["num_detections"],
        "detections": pred["detections"],
        "annotated_image_base64": pred["annotated_image_base64"],
        "latency_ms": pred["latency_ms"],
        "image_shape": pred["image_shape"],
    }

@app.post("/api/v1/analyze")
async def analyze(file: UploadFile = File(...), conf: float = 0.25):
    name = file.filename or ""
    ext = ("." + name.split(".")[-1].lower()) if "." in name else ""
    if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
        raise HTTPException(status_code=400, detail=f"Invalid file extension '{ext}'.")
    content = await file.read()
    cls_pred = classifier.predict(content)
    det_pred = detector.predict(content, conf_threshold=conf, include_base64_image=True)
    return {
        "status": "success",
        "classification": {
            "status": "success",
            "predicted_class": cls_pred["predicted_class"],
            "display_name": cls_pred["display_name"],
            "confidence_pct": cls_pred["confidence"],
            "probabilities": cls_pred["probabilities"],
            "latency_ms": cls_pred["latency_ms"],
            "image_shape": cls_pred["image_shape"],
        },
        "detection": {
            "status": "success",
            "num_detections": det_pred["num_detections"],
            "detections": det_pred["detections"],
            "annotated_image_base64": det_pred["annotated_image_base64"],
            "latency_ms": det_pred["latency_ms"],
            "image_shape": det_pred["image_shape"],
        },
        "total_latency_ms": round(cls_pred["latency_ms"] + det_pred["latency_ms"], 2),
    }
