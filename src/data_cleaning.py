"""
data_cleaning.py — Fonctions de chargement et nettoyage des données
Projet Capstone : Prédiction des Prix Immobiliers à Nouakchott
"""

import pandas as pd
import numpy as np


def load_data(train_path: str, test_path: str):
    """Charge les fichiers CSV train et test."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def clean_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Nettoyage de base :
    - Conversion date_publication en datetime
    - Strip des noms de quartiers
    - Vérification des prix (train uniquement)
    """
    df = df.copy()

    # Standardiser les noms de quartiers
    df['quartier'] = df['quartier'].str.strip()

    # Convertir date_publication en datetime
    df['date_publication'] = pd.to_datetime(df['date_publication'], errors='coerce')

    if is_train:
        # Vérifier la plage des prix
        assert df['prix'].min() >= 100_000, "Prix inférieurs à 100K MRU trouvés"
        assert df['prix'].max() <= 100_000_000, "Prix supérieurs à 100M MRU trouvés"

    # Surface cohérente (> 0)
    df = df[df['surface_m2'] > 0].copy()

    return df


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne un tableau résumant les valeurs manquantes."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({
        'nb_manquant': missing,
        'pct_manquant': pct
    }).sort_values('pct_manquant', ascending=False)


def get_basic_stats(df: pd.DataFrame) -> dict:
    """Retourne les statistiques de base du dataset."""
    return {
        'shape': df.shape,
        'dtypes': df.dtypes,
        'describe': df.describe(),
        'head': df.head(),
        'quartiers': df['quartier'].value_counts(),
        'date_range': (df['date_publication'].min(), df['date_publication'].max()),
        'sources': df['source'].value_counts() if 'source' in df.columns else None,
    }
