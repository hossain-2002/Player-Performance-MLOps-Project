"""Training script for predicting `overall_rating`.

This script loads preprocessed numpy arrays, trains three models (LinearRegression,
RandomForestRegressor, XGBoostRegressor) each in separate MLflow runs, logs
parameters/metrics/models, selects the best model by RMSE and saves it.
"""

import os
import pickle
from typing import Dict, Tuple

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb


def load_data(data_dir: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load numpy arrays X_train, X_test, y_train, y_test from the data directory.

    Args:
        data_dir: path to data directory (defaults to repo `data/` folder)

    Returns:
        X_train, X_test, y_train, y_test
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Predict with `model` and return RMSE, MAE, R2 metrics.

    Args:
        model: fitted model with a .predict()
        X_test: test features
        y_test: test targets

    Returns:
        dict with keys 'rmse', 'mae', 'r2'
    """
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def train_and_log(model, name: str, params: dict, X_train, X_test, y_train, y_test) -> Tuple[object, dict]:
    """Train `model`, log params/metrics/model to MLflow under a dedicated run.

    Args:
        model: an unfitted estimator instance
        name: run name / model name
        params: dictionary of parameters to log
        X_train, X_test, y_train, y_test: datasets

    Returns:
        fitted model and metrics dict
    """
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        # fit
        model.fit(X_train, y_train)
        # evaluate
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # log model artifact
        if name.lower().startswith('xgboost'):
            mlflow.xgboost.log_model(model, artifact_path='model')
        else:
            mlflow.sklearn.log_model(model, artifact_path='model')

    return model, metrics


def main():
    """Main training flow: loads data, sets MLflow config, trains models, selects best.
    """
    X_train, X_test, y_train, y_test = load_data()

    # 2) MLflow setup
    from pathlib import Path
    mlruns_dir = Path(os.path.join(os.path.dirname(__file__), '..', 'mlruns')).resolve()
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    mlflow.set_experiment('player-performance')

    models_to_run = [
        ('LinearRegression', LinearRegression(), {'fit_intercept': True}),
        ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42), {'n_estimators': 100, 'random_state': 42}),
        ('XGBoost', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0), {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}),
    ]

    results = {}
    fitted_models = {}

    for name, model, params in models_to_run:
        print(f"Training {name}...")
        fitted, metrics = train_and_log(model, name, params, X_train, X_test, y_train, y_test)
        results[name] = metrics
        fitted_models[name] = fitted

    # 5) Determine best model by RMSE
    best_name = min(results.keys(), key=lambda n: results[n]['rmse'])
    best_metrics = results[best_name]
    print('\nModel RMSEs:')
    for n, m in results.items():
        print(f"- {n}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R2={m['r2']:.4f}")

    print(f"\nBest model by RMSE: {best_name} (RMSE={best_metrics['rmse']:.4f})")

    # 6) Save best model as pickle
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    best_model = fitted_models[best_name]
    with open(os.path.join(models_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)


if __name__ == '__main__':
    main()

