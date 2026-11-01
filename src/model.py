import tensorflow as tf
from tensorflow.keras import layers
from ultralytics import YOLO
from src.config import CLASSIFICATION_MODEL_PATH, DETECTION_MODEL_PATH

def build_googlenet_model():
    base = tf.keras.applications.InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    model = tf.keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(4, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def load_classifier_model(path=None):
    return tf.keras.models.load_model(str(path or CLASSIFICATION_MODEL_PATH))

def load_detector_model(path=None):
    return YOLO(str(path or DETECTION_MODEL_PATH))
