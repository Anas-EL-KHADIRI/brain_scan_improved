# 🧠 BrainScan AI - Analyse Automatisée d'Images Cérébrales

[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Clinical_UI-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-blueviolet.svg)](https://ultralytics.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-GoogLeNet_Keras-orange.svg)](https://www.tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)

Solution complète et unifiée d'intelligence artificielle clinique pour l'analyse automatisée de clichés IRM du cerveau.
Le projet combine deux architectures complémentaires de Deep Learning :
1. **Classification des tumeurs cérébrales** avec **GoogLeNet (InceptionV1)**.
2. **Détection et localisation des masses tumorales** (boîtes englobantes) avec **YOLOv8**.

---

## 🏛️ Architecture du Projet

```text
brain_scan_improved/
│
├── api/                             # Service REST FastAPI
│   ├── main.py                      # Endpoints (/health, /classify, /detect, /analyze)
│   └── schemas.py                   # Schémas de validation Pydantic
│
├── app/                             # Interface Utilisateur Streamlit
│   └── app.py                       # Dashboard clinique interactif
│
├── airflow/                         # Automatisation & Orchestration
│   └── dags/
│       └── brain_tumor_pipeline_dag.py
│
├── data/                            # Données IRM
│   ├── data.yaml                    # Configuration YOLO des classes & chemins
│   ├── raw/                         # Images brutes par classe (glioma, meningioma, ...)
│   └── yolo/                        # Splits YOLO (images/ & labels/)
│
├── models/                          # Poids des modèles entraînés
│   ├── googlenet_brain_tumor.keras  # Modèle Inception Keras
│   └── yolov8_brain_tumor.pt        # Modèle YOLOv8
│
├── src/                             # Moteur Python Modulaire
│   ├── config.py                    # Chemins, hyperparamètres et constantes
│   ├── data_loader.py               # Validation intégrité, augmentation et chargement
│   ├── model.py                     # Définition des architectures & transfer learning
│   ├── predict.py                   # Moteur d'inférence haute performance
│   ├── train.py                     # Script d'entraînement avec suivi MLflow
│   ├── evaluate.py                  # Évaluation des métriques et matrice de confusion
│   └── visualization.py             # Visualisations & graphiques médicaux
│
├── tests/                           # Suite de tests automatisés (Pytest)
│   ├── test_api.py
│   ├── test_model.py
│   └── test_data_loader.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 Démarrage Rapide

### 1. Activer l'Environnement Virtuel

```powershell
# Sous Windows PowerShell :
.\.venv\Scripts\Activate.ps1
```

### 2. Démarrer l'API REST FastAPI

Le backend expose la documentation interactive OpenAPI (Swagger) sur le port `8000` :

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Swagger UI** : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check** : [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

### 3. Lancer le Dashboard Clinique Streamlit

Dans un second terminal, démarrez l'interface web pour les praticiens :

```powershell
streamlit run app/app.py
```

Accédez à l'application sur [http://localhost:8501](http://localhost:8501).

---

## 📡 Endpoints de l'API REST

| Méthode | Route | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Informations générales et liste des routes |
| `GET` | `/api/v1/health` | Statut de disponibilité des modèles en mémoire |
| `POST` | `/api/v1/classify` | Inférence GoogLeNet (diagnostic + probabilités) |
| `POST` | `/api/v1/detect` | Inférence YOLOv8 (coordonnées boîtes + image annotée) |
| `POST` | `/api/v1/analyze` | Analyse complète conjointe (GoogLeNet + YOLOv8) |

---

## 🧪 Exécution des Tests

Exécutez l'ensemble de la suite de tests unitaires et d'intégration :

```powershell
pytest tests/ -v
```

---

## 📊 Suivi des Expériences avec MLflow

Pour consulter les métriques d'entraînement, les hyperparamètres et les artefacts :

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Ouvrez [http://127.0.0.1:5000](http://127.0.0.1:5000) dans votre navigateur.
