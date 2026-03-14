"""
modeling.py — Entraînement, évaluation et comparaison des modèles
Projet Capstone : Prédiction des Prix Immobiliers à Nouakchott
"""

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ── Métriques ─────────────────────────────────────────────────────────────────

def rmsle_original_scale(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """RMSLE recalculé en revenant à l'échelle originale."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    # Clip pour éviter log de 0
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ── Baseline ──────────────────────────────────────────────────────────────────

def baseline_rmsle(y_train_log: np.ndarray, y_val_log: np.ndarray) -> float:
    """Prédit la médiane du train pour tous : baseline."""
    median_pred = np.full_like(y_val_log, np.median(y_train_log))
    return rmsle_original_scale(y_val_log, median_pred)


# ── Définition des modèles ────────────────────────────────────────────────────

def get_models() -> dict:
    """Retourne le dictionnaire des modèles à comparer."""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge':             Ridge(alpha=1.0),
        'Lasso':             Lasso(alpha=0.001, max_iter=10000),
        'Random Forest':     RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=42
        ),
    }

    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0
        )

    if HAS_LGB:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        )

    return models


# ── Validation croisée ────────────────────────────────────────────────────────

def cross_validate_model(model, X: np.ndarray, y: np.ndarray,
                          n_splits: int = 5) -> dict:
    """
    KFold cross-validation (5-fold).
    Retourne les métriques moyennées : RMSE, MAE, R², RMSLE.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rmses, maes, r2s, rmsles = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        rmses.append(rmse(y_val, y_pred))
        maes.append(mean_absolute_error(y_val, y_pred))
        r2s.append(r2_score(y_val, y_pred))
        rmsles.append(rmsle_original_scale(y_val, y_pred))

    return {
        'RMSE_log':  np.mean(rmses),
        'MAE_log':   np.mean(maes),
        'R2':        np.mean(r2s),
        'RMSLE':     np.mean(rmsles),
        'RMSE_std':  np.std(rmses),
        'RMSLE_std': np.std(rmsles),
    }


def compare_all_models(X: np.ndarray, y: np.ndarray,
                        baseline_rmsle_val: float) -> pd.DataFrame:
    """
    Entraîne et évalue tous les modèles en cross-validation.
    Retourne un DataFrame de comparaison, trié par RMSLE.
    """
    models = get_models()
    results = []

    # Baseline
    results.append({
        'Modèle': 'Baseline (médiane)',
        'RMSE_log': np.nan,
        'MAE_log':  np.nan,
        'R2':       np.nan,
        'RMSLE':    baseline_rmsle_val,
        'RMSE_std': np.nan,
        'RMSLE_std': np.nan,
    })

    for name, model in models.items():
        print(f"  → Entraînement {name}...")
        metrics = cross_validate_model(model, X, y)
        metrics['Modèle'] = name
        results.append(metrics)

    df_results = pd.DataFrame(results).set_index('Modèle')
    return df_results.sort_values('RMSLE')


# ── Entraînement final + sauvegarde ──────────────────────────────────────────

def train_final_model(model, X: np.ndarray, y: np.ndarray):
    """Entraîne le modèle sur l'intégralité du train."""
    model.fit(X, y)
    return model


def save_model(model, feature_cols: list,
               model_dir: str = 'model/') -> None:
    """Sérialise le modèle et la liste de features."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model,        os.path.join(model_dir, 'housing_model.pkl'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'features.pkl'))
    print(f"Modèle sauvegardé dans {model_dir}")


def load_model(model_dir: str = 'model/'):
    """Charge le modèle et les features depuis le disque."""
    model        = joblib.load(os.path.join(model_dir, 'housing_model.pkl'))
    feature_cols = joblib.load(os.path.join(model_dir, 'features.pkl'))
    return model, feature_cols


# ── Génération de la soumission ───────────────────────────────────────────────

def generate_submission(model, X_test: np.ndarray, test_ids: pd.Series,
                         output_path: str = 'data/final/submission.csv') -> pd.DataFrame:
    """Génère le fichier de soumission Kaggle."""
    log_preds = model.predict(X_test)
    preds = np.maximum(np.expm1(log_preds), 100_000)

    submission = pd.DataFrame({
        'id':   test_ids.values,
        'prix': preds.round(0).astype(int),
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Soumission sauvegardée : {output_path} ({len(submission)} lignes)")

    assert len(submission) == 289, f"Attendu 289 lignes, obtenu {len(submission)}"
    assert submission['prix'].isna().sum() == 0, "NaN dans les prédictions !"

    return submission
