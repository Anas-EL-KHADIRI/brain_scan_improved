import mlflow
from tensorflow.keras.utils import to_categorical
from src.config import CLASSIFICATION_CLASSES, CLASSIFICATION_EXPERIMENT_NAME, CLASSIFICATION_MODEL_PATH, MLFLOW_TRACKING_URI, MODELS_DIR, PROCESSED_DATA_DIR
from src.data_loader import load_dataset_arrays, split_classification_data
from src.model import build_googlenet_model

def train():
    train_dir = PROCESSED_DATA_DIR / "classification" / "train"
    val_dir = PROCESSED_DATA_DIR / "classification" / "val"
    if not train_dir.exists():
        split_classification_data()

    X_train, y_train = load_dataset_arrays(train_dir)
    X_val, y_val = load_dataset_arrays(val_dir)

    y_train = to_categorical(y_train, len(CLASSIFICATION_CLASSES))
    y_val = to_categorical(y_val, len(CLASSIFICATION_CLASSES))

    model = build_googlenet_model()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(CLASSIFICATION_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="GoogLeNet_Training"):
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
        mlflow.log_metrics({
            "train_loss": history.history["loss"][-1],
            "train_acc": history.history["accuracy"][-1],
            "val_loss": history.history["val_loss"][-1],
            "val_acc": history.history["val_accuracy"][-1],
        })
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(str(CLASSIFICATION_MODEL_PATH))
        mlflow.log_artifact(str(CLASSIFICATION_MODEL_PATH))

if __name__ == "__main__":
    train()
