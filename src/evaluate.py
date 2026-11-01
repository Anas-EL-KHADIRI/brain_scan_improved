import mlflow
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.config import CLASSIFICATION_CLASSES, CLASSIFICATION_EXPERIMENT_NAME, CLASSIFICATION_MODEL_PATH, FIGURES_DIR, MLFLOW_TRACKING_URI, PROCESSED_DATA_DIR
from src.data_loader import load_dataset_arrays
from src.model import load_classifier_model
from src.visualization import plot_confusion_matrix_heatmap

def evaluate():
    test_dir = PROCESSED_DATA_DIR / "classification" / "test"
    if not test_dir.exists():
        return

    X_test, y_test = load_dataset_arrays(test_dir)
    model = load_classifier_model(CLASSIFICATION_MODEL_PATH)
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    rep = classification_report(y_test, y_pred, target_names=CLASSIFICATION_CLASSES, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    cm_path = FIGURES_DIR / "evaluation_confusion_matrix.png"
    plot_confusion_matrix_heatmap(cm, CLASSIFICATION_CLASSES, save_path=cm_path)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(CLASSIFICATION_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="GoogLeNet_Evaluation"):
        mlflow.log_metrics({
            "test_accuracy": rep["accuracy"],
            "macro_f1": rep["macro avg"]["f1-score"],
            "weighted_precision": rep["weighted avg"]["precision"],
            "weighted_recall": rep["weighted avg"]["recall"],
        })
        mlflow.log_artifact(str(cm_path))

if __name__ == "__main__":
    evaluate()
