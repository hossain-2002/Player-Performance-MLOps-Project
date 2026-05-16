# ⚽ Player Performance Prediction — MLOps Pipeline

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?logo=dvc&logoColor=white)

End-to-end MLOps pipeline predicting EA FC25 player overall ratings using machine learning with full experiment tracking, API serving, and CI/CD.

## Architecture

```text
Data → Preprocess → Train (MLflow) → Best Model → FastAPI → Docker
  │         │            │               │             │        │
  └─────────┴────────────┴───────────────┴─────────────┴────────┴──→ Deploy
```

## Tech Stack

| Category | Tool |
|---|---|
| ML | `scikit-learn`, `xgboost`, `numpy`, `pandas` |
| API | `FastAPI`, `Uvicorn`, `Pydantic` |
| Containerization | `Docker`, `docker-compose` |
| Experiment Tracking | `MLflow` |
| Data Versioning | `DVC` |
| CI/CD | `GitHub Actions` |

## Project Structure

```text
player-performance-mlops/
├── api/
│   └── main.py
├── data/
│   ├── new-players-data-full.csv
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── le_best_position.pkl
│   └── le_preferred_foot.pkl
├── mlruns/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── preprocess.py
│   └── train.py
├── .github/
│   └── workflows/
│       └── train.yml
├── .dvcignore
├── dvc.yaml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quick Start

### 1) Clone the repository

```bash
git clone https://github.com/hossain-2002/Player-Performance-MLOps-Project.git
cd player-performance-mlops
```

### 2) Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Run preprocessing

```bash
python src/preprocess.py
```

### 4) Train models

```bash
python src/train.py
```

### 5) Start the API with Docker

```bash
docker compose up --build api
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Returns a simple service status message. |
| `GET` | `/health` | Checks whether the model and scaler are loaded. |
| `POST` | `/predict` | Predicts a player’s `overall_rating` from input features. |

## Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "height_cm": 180,
    "weight_kg": 75,
    "age": 24,
    "pace": 80,
    "shooting": 70,
    "passing": 72,
    "dribbling": 78,
    "defending": 60,
    "physic": 75,
    "value_eur": 5000000,
    "wage_eur": 20000,
    "best_position_encoded": 5,
    "preferred_foot_encoded": 1
  }'
```

Example response:

```json
{
  "predicted_overall_rating": 82.4
}
```

## MLflow

MLflow logs experiment runs locally in the `mlruns/` directory.

To view experiments in the MLflow UI:

```bash
mlflow ui --backend-store-uri mlruns
```

Then open:

```text
http://127.0.0.1:5000
```

## Results

Model comparison results will be logged here after training.

| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Linear Regression | 1.9142 | 1.4838 | 0.9228 |
| Random Forest Regressor | 0.7945 | 0.5204 | 0.9867 |
| XGBoost Regressor | 0.7390 | 0.5465 | 0.9885 |

## License

This project is licensed under the MIT License.
