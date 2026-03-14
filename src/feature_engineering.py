"""
feature_engineering.py — Extraction de features textuelles, temporelles et dérivées
Projet Capstone : Prédiction des Prix Immobiliers à Nouakchott
"""

import re
import pandas as pd
import numpy as np

DATE_REF = pd.Timestamp('2026-03-02')

# Liste complète des 34 features
FEATURE_COLS = [
    'surface_m2', 'nb_chambres', 'nb_salons', 'nb_sdb', 'has_sdb_info',
    'latitude', 'longitude', 'dist_centre_km', 'dist_aeroport_km', 'dist_plage_km',
    'nb_ecoles_1km', 'nb_mosquees_1km', 'nb_commerces_1km', 'nb_hopitaux_1km', 'nb_total_pois_1km',
    'nb_pieces_total', 'has_garage', 'has_titre_foncier', 'has_camera', 'nb_balcons', 'taille_rue',
    'desc_len', 'desc_word_count',
    'has_piscine', 'has_clim', 'has_meuble', 'is_luxe', 'is_renove', 'has_arabic',
    'age_annonce_jours', 'surface_par_piece', 'log_surface',
    'quartier_target_enc', 'quartier_freq'
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_str(val) -> str:
    """Convertit en string en gérant les NaN."""
    if pd.isna(val):
        return ''
    return str(val).lower()


def _extract_caract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait des features depuis la colonne 'caracteristiques'."""
    df = df.copy()

    caract = df['caracteristiques'].fillna('').str.lower()

    df['has_garage']        = caract.str.contains('garage', na=False).astype(int)
    df['has_titre_foncier'] = caract.str.contains('titre foncier', na=False).astype(int)
    df['has_camera']        = caract.str.contains('cam', na=False).astype(int)

    # Nombre de balcons
    def _extract_balcons(s):
        m = re.search(r'(\d+)\s*balcon', s)
        return int(m.group(1)) if m else 0

    df['nb_balcons'] = caract.apply(_extract_balcons)

    # Taille de rue
    def _extract_taille_rue(s):
        m = re.search(r'taille rue:\s*([\d.]+)', s)
        if m:
            try:
                val = float(m.group(1))
                return val if val > 0 else np.nan
            except ValueError:
                return np.nan
        return np.nan

    df['taille_rue'] = caract.apply(_extract_taille_rue)

    return df


def _extract_description_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait des features NLP basiques depuis 'description' et 'caracteristiques'."""
    df = df.copy()

    desc    = df['description'].fillna('')
    caract  = df['caracteristiques'].fillna('')
    combined = (desc + ' ' + caract).str.lower()

    df['desc_len']        = desc.str.len()
    df['desc_word_count'] = desc.str.split().str.len().fillna(0)

    df['has_piscine'] = combined.str.contains('piscine', na=False).astype(int)
    df['has_clim']    = combined.str.contains(r'climatisation|clim', na=False, regex=True).astype(int)
    df['has_meuble']  = combined.str.contains('meubl', na=False).astype(int)
    df['is_luxe']     = combined.str.contains(r'luxe|haut standing|standing', na=False, regex=True).astype(int)
    df['is_renove']   = combined.str.contains(r'rénov|renov|neuf|nouveau', na=False, regex=True).astype(int)

    # Détection de caractères arabes
    df['has_arabic'] = df['description'].fillna('').apply(
        lambda s: int(bool(re.search(r'[\u0600-\u06FF]', s)))
    )

    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features dérivées (pièces, surface, âge)."""
    df = df.copy()

    df['nb_pieces_total']   = df['nb_chambres'].fillna(0) + df['nb_salons'].fillna(0)
    df['surface_par_piece'] = df['surface_m2'] / df['nb_pieces_total'].clip(lower=1)
    df['log_surface']       = np.log1p(df['surface_m2'])

    if 'date_publication' in df.columns:
        df['age_annonce_jours'] = (DATE_REF - df['date_publication']).dt.days
    else:
        df['age_annonce_jours'] = np.nan

    return df


def add_sdb_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Crée l'indicatrice has_sdb_info (1 si nb_sdb était renseigné, 0 sinon)."""
    df = df.copy()
    df['has_sdb_info'] = df['nb_sdb'].notna().astype(int)
    return df


# ── Imputation (fit sur train, apply sur train+test) ──────────────────────────

class ImputationConfig:
    """Stocke les valeurs d'imputation calculées sur le train."""

    def __init__(self):
        self.chambres_by_quartier: dict = {}
        self.salons_by_quartier: dict   = {}
        self.sdb_median: float          = np.nan
        self.taille_rue_median: float   = np.nan
        self.feature_medians: dict      = {}

    def fit(self, train: pd.DataFrame) -> 'ImputationConfig':
        """Calcule toutes les valeurs d'imputation sur le train."""
        self.chambres_by_quartier = (
            train.groupby('quartier')['nb_chambres']
            .median()
            .to_dict()
        )
        self.salons_by_quartier = (
            train.groupby('quartier')['nb_salons']
            .median()
            .to_dict()
        )
        self.sdb_median      = train['nb_sdb'].median()
        self.taille_rue_median = np.nan  # sera calculée après extraction
        return self

    def fit_taille_rue(self, train: pd.DataFrame) -> 'ImputationConfig':
        """Calcule la médiane de taille_rue après extraction."""
        self.taille_rue_median = train['taille_rue'].median()
        return self

    def fit_feature_medians(self, train: pd.DataFrame) -> 'ImputationConfig':
        """Calcule les médianes pour toutes les features numériques (fill-in final)."""
        for col in FEATURE_COLS:
            if col in train.columns:
                med = train[col].median()
                self.feature_medians[col] = med if not pd.isna(med) else 0.0
        return self

    def apply_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique l'imputation de base."""
        df = df.copy()

        # nb_chambres par quartier
        def _impute_chambres(row):
            if pd.isna(row['nb_chambres']):
                return self.chambres_by_quartier.get(row['quartier'],
                       pd.Series(self.chambres_by_quartier).median())
            return row['nb_chambres']

        df['nb_chambres'] = df.apply(_impute_chambres, axis=1)

        # nb_salons par quartier
        def _impute_salons(row):
            if pd.isna(row['nb_salons']):
                return self.salons_by_quartier.get(row['quartier'],
                       pd.Series(self.salons_by_quartier).median())
            return row['nb_salons']

        df['nb_salons'] = df.apply(_impute_salons, axis=1)

        # nb_sdb par médiane globale
        df['nb_sdb'] = df['nb_sdb'].fillna(self.sdb_median)

        # caracteristiques : NaN → chaîne vide
        df['caracteristiques'] = df['caracteristiques'].fillna('')

        return df

    def apply_taille_rue(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute taille_rue manquant par la médiane calculée sur le train."""
        df = df.copy()
        df['taille_rue'] = df['taille_rue'].fillna(self.taille_rue_median)
        return df

    def apply_feature_medians(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remplit tous les NaN restants dans les features par la médiane du train."""
        df = df.copy()
        for col, med in self.feature_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(med)
        return df


# ── Encodage du quartier (target encoding) ────────────────────────────────────

class QuartierEncoder:
    """Target encoding + fréquence, ajusté sur le train uniquement."""

    def __init__(self):
        self.target_enc: dict = {}
        self.freq_enc: dict   = {}

    def fit(self, train: pd.DataFrame, target_col: str = 'prix') -> 'QuartierEncoder':
        self.target_enc = train.groupby('quartier')[target_col].mean().to_dict()
        freq = train['quartier'].value_counts(normalize=True)
        self.freq_enc = freq.to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        global_mean = pd.Series(self.target_enc).mean()
        df['quartier_target_enc'] = df['quartier'].map(self.target_enc).fillna(global_mean)
        df['quartier_freq']       = df['quartier'].map(self.freq_enc).fillna(0.0)
        return df


# ── Pipeline principal ────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame,
                   imputer: ImputationConfig,
                   quartier_enc: QuartierEncoder,
                   is_train: bool = True) -> pd.DataFrame:
    """
    Pipeline complet de feature engineering sur un dataframe déjà nettoyé
    et géo-enrichi.
    """
    df = imputer.apply_basic(df)
    df = add_sdb_indicator(df)          # doit venir APRÈS imputation sdb
    df = _extract_caract_features(df)
    df = imputer.apply_taille_rue(df)
    df = _extract_description_features(df)
    df = _add_derived_features(df)
    df = quartier_enc.transform(df)

    # Fill-in final
    df = imputer.apply_feature_medians(df)

    return df
