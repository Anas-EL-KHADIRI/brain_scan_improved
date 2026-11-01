import base64
from pathlib import Path
import time
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from src.config import CLASSIFICATION_CLASSES, CLASSIFICATION_MODEL_PATH, DETECTION_MODEL_PATH

def read_image(image_input):
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(image_input, bytes):
        arr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(image_input)

class BrainClassifier:
    def __init__(self, model_path=None):
        self.model = tf.keras.models.load_model(str(model_path or CLASSIFICATION_MODEL_PATH))
        self.classes = CLASSIFICATION_CLASSES

    def predict(self, image_input):
        t0 = time.time()
        img = read_image(image_input)
        resized = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        batch = np.expand_dims(resized, axis=0)

        probs = self.model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))

        return {
            "predicted_class": self.classes[idx],
            "display_name": self.classes[idx],
            "confidence": round(float(probs[idx]) * 100, 2),
            "probabilities": {c: round(float(p) * 100, 2) for c, p in zip(self.classes, probs)},
            "latency_ms": round((time.time() - t0) * 1000, 2),
            "image_shape": [img.shape[0], img.shape[1]],
        }

class BrainTumorDetector:
    def __init__(self, model_path=None):
        self.model = YOLO(str(model_path or DETECTION_MODEL_PATH))

    def predict(self, image_input, conf_threshold=0.25, include_base64_image=True):
        t0 = time.time()
        img = read_image(image_input)
        res = self.model.predict(img, conf=conf_threshold, verbose=False)[0]

        detections = []
        for i, box in enumerate(res.boxes):
            cid = int(box.cls[0].item())
            name = self.model.names.get(cid, str(cid))
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0].tolist()]
            detections.append({
                "box_id": i + 1,
                "class_id": cid,
                "class_name": name,
                "display_name": name,
                "confidence": round(conf * 100, 2),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

        b64 = None
        if include_base64_image:
            annotated = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            b64 = base64.b64encode(buf).decode("utf-8")

        return {
            "num_detections": len(detections),
            "detections": detections,
            "annotated_image_base64": b64,
            "latency_ms": round((time.time() - t0) * 1000, 2),
            "image_shape": [img.shape[0], img.shape[1]],
        }
