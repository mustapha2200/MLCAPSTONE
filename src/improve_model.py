#!/usr/bin/env python3
"""
improve_model.py — Objectif RMSLE < 0.57
Pipeline amélioré : NLP avancé + interactions + XGBoost/LightGBM/CatBoost + Stacking
"""

import sys, os, re, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
import json

ROOT      = os.path.join(os.path.dirname(__file__), '..')
TRAIN_CSV = os.path.join(ROOT, 'data', 'raw', 'kaggle_train.csv')
TEST_CSV  = os.path.join(ROOT, 'data', 'raw', 'kaggle_test.csv')
OUT_DIR   = os.path.join(ROOT, 'data', 'final')
MODEL_DIR = os.path.join(ROOT, 'model')
GEO_META  = os.path.join(ROOT, 'data', 'processed', 'geo_meta.json')
DATE_REF  = pd.Timestamp('2026-03-02')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Charger les données géographiques depuis le fichier JSON (aucune valeur hardcodée)
with open(GEO_META, encoding='utf-8') as f:
    _geo = json.load(f)

QUARTIER_META = _geo['quartiers']
QUARTIER_GPS  = {q: (v['lat'], v['lon']) for q, v in QUARTIER_META.items()}
QUARTIER_POI  = {
    q: (v['poi']['ecoles'], v['poi']['mosquees'], v['poi']['commerces'], v['poi']['hopitaux'])
    for q, v in QUARTIER_META.items()
}
_ref          = _geo['reference_points']
CENTRE_GPS    = (_ref['centre']['lat'],   _ref['centre']['lon'])
AEROPORT_GPS  = (_ref['aeroport']['lat'], _ref['aeroport']['lon'])
PLAGE_GPS     = (_ref['plage']['lat'],    _ref['plage']['lon'])


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────
def rmsle(y_true_log, y_pred_log):
    y_true = np.expm1(np.clip(y_true_log, 0, None))
    y_pred = np.expm1(np.clip(y_pred_log, 0, None))
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def haversine(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 : CHARGEMENT & NETTOYAGE
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ÉTAPE 1 — Chargement & nettoyage")
print("=" * 60)

train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

for df in [train, test]:
    df['quartier'] = df['quartier'].str.strip()
    df['date_publication'] = pd.to_datetime(df['date_publication'], errors='coerce')
    df['caracteristiques'] = df['caracteristiques'].fillna('')
    df['description']      = df['description'].fillna('')
    df['titre']            = df['titre'].fillna('')

print(f"Train : {train.shape} | Test : {test.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 : IMPUTATION (fit sur train uniquement)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 2 — Imputation")
print("=" * 60)

# has_sdb_info AVANT imputation
for df in [train, test]:
    df['has_sdb_info'] = df['nb_sdb'].notna().astype(int)

# Médianes par quartier (train only)
ch_med  = train.groupby('quartier')['nb_chambres'].median()
sal_med = train.groupby('quartier')['nb_salons'].median()
sdb_med = train['nb_sdb'].median()
global_ch_med  = train['nb_chambres'].median()
global_sal_med = train['nb_salons'].median()

for df in [train, test]:
    df['nb_chambres'] = df.apply(
        lambda r: ch_med.get(r['quartier'], global_ch_med) if pd.isna(r['nb_chambres']) else r['nb_chambres'], axis=1)
    df['nb_salons'] = df.apply(
        lambda r: sal_med.get(r['quartier'], global_sal_med) if pd.isna(r['nb_salons']) else r['nb_salons'], axis=1)
    df['nb_sdb'] = df['nb_sdb'].fillna(sdb_med)

print("Imputation OK")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 : FEATURE ENGINEERING COMPLET
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 3 — Feature Engineering")
print("=" * 60)

def build_features(train_df, test_df):
    """
    Construit toutes les features.
    Tout ce qui nécessite un fit (TF-IDF, encodages) est calculé sur train_df
    et appliqué sur les deux.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    dfs = {'train': train_df.copy(), 'test': test_df.copy()}

    # ── A) Texte combiné ────────────────────────────────────────────────────
    for name, df in dfs.items():
        df['text_combined'] = (
            df['titre'] + ' ' + df['description'] + ' ' + df['caracteristiques']
        ).str.lower()

    # ── B) TF-IDF + SVD (fit sur train) ────────────────────────────────────
    print("  TF-IDF + SVD...")
    tfidf = TfidfVectorizer(max_features=300, min_df=2, max_df=0.95, ngram_range=(1, 2))
    tfidf_train = tfidf.fit_transform(dfs['train']['text_combined'])
    tfidf_test  = tfidf.transform(dfs['test']['text_combined'])

    svd = TruncatedSVD(n_components=20, random_state=42)
    svd_train = svd.fit_transform(tfidf_train)
    svd_test  = svd.transform(tfidf_test)

    for i in range(20):
        dfs['train'][f'tfidf_{i}'] = svd_train[:, i]
        dfs['test'][f'tfidf_{i}']  = svd_test[:, i]

    # ── C) Mots-clés FR + AR ────────────────────────────────────────────────
    print("  Mots-clés...")
    keywords = {
        'has_garage':         r'garage|كراج',
        'has_titre_foncier':  r'titre foncier|وثيقة ملكية',
        'has_camera':         r'cam|كاميرا',
        'has_piscine':        r'piscine|مسبح',
        'has_clim':           r'climatisation|clim|مكيف',
        'has_meuble':         r'meubl|مفروش',
        'is_luxe':            r'luxe|haut standing|standing|فاخر',
        'is_renove':          r'rénov|renov|neuf|nouveau|جديد|مجدد',
        'has_etage':          r'étage|etage|طابق|طوابق',
        'has_duplex':         r'duplex|دوبلكس',
        'has_jardin':         r'jardin|حديقة',
        'has_terrasse':       r'terrasse|تراس|سطح',
        'has_cuisine_equip':  r'cuisine.quip|مطبخ مجهز',
        'has_carrelage':      r'carrelage|كارو|caro',
        'has_2_facades':      r'façade|واجه|facade',
        'mention_coin':       r'\bcoin\b|ركن|angle',
        'mention_prix_neg':   r'négociable|قابل للتفاوض|negotiable',
        'mention_urgent':     r'\burgent\b|مستعجل|فرصة',
        'has_arabic':         r'[\u0600-\u06FF]',
        'desc_has_numbers':   r'\d{6,}',
    }

    for name, df in dfs.items():
        combined = df['text_combined']
        for feat, pattern in keywords.items():
            df[feat] = combined.str.contains(pattern, na=False, regex=True).astype(int)
        df['desc_len']        = df['description'].str.len().fillna(0)
        df['desc_word_count'] = df['description'].str.split().str.len().fillna(0)
        df['desc_nb_digits']  = df['description'].str.count(r'\d').fillna(0)

    # ── D) Extraction depuis caracteristiques ────────────────────────────────
    for name, df in dfs.items():
        caract = df['caracteristiques'].str.lower()
        def _balcons(s):
            m = re.search(r'(\d+)\s*balcon', str(s))
            return int(m.group(1)) if m else 0
        def _taille_rue(s):
            m = re.search(r'taille rue:\s*([\d.]+)', str(s))
            return float(m.group(1)) if m else np.nan
        df['nb_balcons'] = caract.apply(_balcons)
        df['taille_rue'] = caract.apply(_taille_rue)

    taille_med = dfs['train']['taille_rue'].median()
    for name, df in dfs.items():
        df['taille_rue'] = df['taille_rue'].fillna(taille_med)

    # ── E) Géo ──────────────────────────────────────────────────────────────
    print("  Geo features...")
    for name, df in dfs.items():
        df['latitude']  = df['quartier'].map({q: v[0] for q, v in QUARTIER_GPS.items()})
        df['longitude'] = df['quartier'].map({q: v[1] for q, v in QUARTIER_GPS.items()})
        df['dist_centre_km']   = df.apply(lambda r: haversine(r['latitude'], r['longitude'], *CENTRE_GPS), axis=1)
        df['dist_aeroport_km'] = df.apply(lambda r: haversine(r['latitude'], r['longitude'], *AEROPORT_GPS), axis=1)
        df['dist_plage_km']    = df.apply(lambda r: haversine(r['latitude'], r['longitude'], *PLAGE_GPS), axis=1)
        df['nb_ecoles_1km']    = df['quartier'].map({q: v[0] for q, v in QUARTIER_POI.items()})
        df['nb_mosquees_1km']  = df['quartier'].map({q: v[1] for q, v in QUARTIER_POI.items()})
        df['nb_commerces_1km'] = df['quartier'].map({q: v[2] for q, v in QUARTIER_POI.items()})
        df['nb_hopitaux_1km']  = df['quartier'].map({q: v[3] for q, v in QUARTIER_POI.items()})
        df['nb_total_pois_1km'] = (df['nb_ecoles_1km'] + df['nb_mosquees_1km'] +
                                   df['nb_commerces_1km'] + df['nb_hopitaux_1km'])

    # ── F) Variables dérivées de base ───────────────────────────────────────
    for name, df in dfs.items():
        df['nb_pieces_total']   = df['nb_chambres'] + df['nb_salons']
        df['surface_par_piece'] = df['surface_m2'] / df['nb_pieces_total'].clip(lower=1)
        df['log_surface']       = np.log1p(df['surface_m2'])
        df['log_chambres']      = np.log1p(df['nb_chambres'])
        df['surface_squared']   = df['surface_m2'] ** 2
        df['densite_pieces']    = df['nb_pieces_total'] / df['surface_m2'].clip(lower=20)
        df['age_annonce_jours'] = (DATE_REF - df['date_publication']).dt.days.fillna(365)
        df['mois_publication']  = df['date_publication'].dt.month.fillna(6)
        df['trimestre']         = df['date_publication'].dt.quarter.fillna(2)
        df['is_weekend']        = (df['date_publication'].dt.dayofweek >= 5).astype(int)

    # ── G) Encodage quartier avancé (fit sur train) ─────────────────────────
    print("  Target encoding avec smoothing...")
    global_mean = dfs['train']['prix'].mean()
    smoothing   = 30

    q_stats = dfs['train'].groupby('quartier')['prix'].agg(['mean', 'count', 'std'])
    q_stats['smoothed'] = (
        (q_stats['count'] * q_stats['mean'] + smoothing * global_mean) /
        (q_stats['count'] + smoothing)
    )
    q_freq = dfs['train']['quartier'].value_counts(normalize=True)

    for name, df in dfs.items():
        df['quartier_target_enc']    = df['quartier'].map(q_stats['mean']).fillna(global_mean)
        df['quartier_target_smooth'] = df['quartier'].map(q_stats['smoothed']).fillna(global_mean)
        df['quartier_freq']          = df['quartier'].map(q_freq).fillna(1/8)
        df['quartier_prix_std']      = df['quartier'].map(q_stats['std']).fillna(q_stats['std'].mean())

    # ── H) Features d'interaction (fit sur train pour les médianes) ─────────
    print("  Features d'interaction...")
    q_ch_med  = dfs['train'].groupby('quartier')['nb_chambres'].median()
    q_sur_med = dfs['train'].groupby('quartier')['surface_m2'].median()

    for name, df in dfs.items():
        df['surface_x_target']    = df['surface_m2'] * df['quartier_target_smooth'] / 1e6
        df['chambres_vs_quartier'] = df['nb_chambres'] - df['quartier'].map(q_ch_med).fillna(3)
        df['surface_vs_quartier']  = df['surface_m2'] - df['quartier'].map(q_sur_med).fillna(200)
        df['log_surface_x_target'] = df['log_surface'] * df['quartier_target_smooth'] / 1e6

    return dfs['train'], dfs['test']


train_fe, test_fe = build_features(train, test)

# ── Winsorizing de la target ──────────────────────────────────────────────────
p1, p99 = train_fe['prix'].quantile(0.01), train_fe['prix'].quantile(0.99)
print(f"\n  Winsorizing : clip prix entre {p1:,.0f} et {p99:,.0f} MRU")
train_fe['prix_w'] = train_fe['prix'].clip(p1, p99)

y      = np.log1p(train_fe['prix'])    # target originale
y_clip = np.log1p(train_fe['prix_w'])  # target winsorisée

# ── Liste des features ────────────────────────────────────────────────────────
TFIDF_COLS = [f'tfidf_{i}' for i in range(20)]

BASE_FEATURES = [
    'surface_m2', 'nb_chambres', 'nb_salons', 'nb_sdb', 'has_sdb_info',
    'latitude', 'longitude', 'dist_centre_km', 'dist_aeroport_km', 'dist_plage_km',
    'nb_ecoles_1km', 'nb_mosquees_1km', 'nb_commerces_1km', 'nb_hopitaux_1km', 'nb_total_pois_1km',
    'nb_pieces_total', 'has_garage', 'has_titre_foncier', 'has_camera', 'nb_balcons', 'taille_rue',
    'desc_len', 'desc_word_count', 'desc_has_numbers', 'desc_nb_digits',
    'has_piscine', 'has_clim', 'has_meuble', 'is_luxe', 'is_renove', 'has_arabic',
    'has_etage', 'has_duplex', 'has_jardin', 'has_terrasse', 'has_carrelage',
    'mention_coin', 'mention_prix_neg', 'mention_urgent', 'has_2_facades',
    'age_annonce_jours', 'mois_publication', 'trimestre', 'is_weekend',
    'surface_par_piece', 'log_surface', 'log_chambres', 'surface_squared', 'densite_pieces',
    'quartier_target_enc', 'quartier_target_smooth', 'quartier_freq', 'quartier_prix_std',
    'surface_x_target', 'chambres_vs_quartier', 'surface_vs_quartier', 'log_surface_x_target',
] + TFIDF_COLS

# S'assurer que toutes les features existent
FEATURE_COLS = [c for c in BASE_FEATURES if c in train_fe.columns]
print(f"\n  Total features : {len(FEATURE_COLS)}")

# Remplir les NaN restants
feature_medians = train_fe[FEATURE_COLS].median()
train_fe[FEATURE_COLS] = train_fe[FEATURE_COLS].fillna(feature_medians)
test_fe[FEATURE_COLS]  = test_fe[FEATURE_COLS].fillna(feature_medians)

X_train = train_fe[FEATURE_COLS]
X_test  = test_fe[FEATURE_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 : VALIDATION CROISÉE DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 4 — Comparaison des modèles (CV 5-fold)")
print("=" * 60)

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}


def cv_score(model, X, y, name):
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        scores.append(rmsle(y_val.values, pred))
    mean_rmsle = np.mean(scores)
    std_rmsle  = np.std(scores)
    print(f"  {name:<30} RMSLE = {mean_rmsle:.4f} ± {std_rmsle:.4f}")
    results[name] = mean_rmsle
    return mean_rmsle


# Ridge baseline
ridge = Ridge(alpha=1.0)
cv_score(ridge, X_train, y, "Ridge")

# Random Forest
rf = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3,
                           max_features=0.6, random_state=42, n_jobs=-1)
cv_score(rf, X_train, y, "Random Forest")

# XGBoost
try:
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5, gamma=0.1,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    cv_score(xgb_model, X_train, y, "XGBoost")
except ImportError:
    print("  XGBoost non disponible")

# LightGBM
try:
    import lightgbm as lgb
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_samples=10,
        num_leaves=63, random_state=42, n_jobs=-1, verbose=-1,
    )
    cv_score(lgb_model, X_train, y, "LightGBM")
except ImportError:
    print("  LightGBM non disponible")

# CatBoost
try:
    from catboost import CatBoostRegressor
    cat_model = CatBoostRegressor(
        iterations=1000, depth=6, learning_rate=0.03,
        l2_leaf_reg=3, random_seed=42, verbose=0,
    )
    cv_score(cat_model, X_train, y, "CatBoost")
except ImportError:
    print("  CatBoost non disponible")

# Gradient Boosting (sklearn)
gbm = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                 min_samples_leaf=5, subsample=0.8, random_state=42)
cv_score(gbm, X_train, y, "GradientBoosting")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 : STACKING / BLENDING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 5 — Stacking OOF")
print("=" * 60)

def get_oof_predictions(model, X, y, X_test, n_folds=5):
    kf_ = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_train = np.zeros(len(X))
    oof_test_folds = np.zeros((len(X_test), n_folds))
    for i, (tr_idx, val_idx) in enumerate(kf_.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr = y.iloc[tr_idx]
        model.fit(X_tr, y_tr)
        oof_train[val_idx]    = model.predict(X_val)
        oof_test_folds[:, i]  = model.predict(X_test)
    return oof_train, oof_test_folds.mean(axis=1)


# Sélectionner les 3 meilleurs modèles
sorted_models = sorted(results.items(), key=lambda x: x[1])
print(f"  Top modèles : {[m for m, _ in sorted_models[:3]]}")

# Construire les candidats disponibles
candidates = {}
if 'XGBoost' in results:
    candidates['xgb'] = xgb.XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=5, gamma=0.1, random_state=42, n_jobs=-1, verbosity=0)
if 'LightGBM' in results:
    candidates['lgb'] = lgb.LGBMRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        min_child_samples=10, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)
if 'CatBoost' in results:
    candidates['cat'] = CatBoostRegressor(
        iterations=1000, depth=6, learning_rate=0.03,
        l2_leaf_reg=3, random_seed=42, verbose=0)
candidates['rf'] = RandomForestRegressor(
    n_estimators=300, max_depth=15, min_samples_leaf=3,
    max_features=0.6, random_state=42, n_jobs=-1)
candidates['ridge'] = Ridge(alpha=1.0)

oof_preds  = {}
test_preds = {}

for name, model in candidates.items():
    print(f"  OOF {name}...")
    oof, test_pred = get_oof_predictions(model, X_train, y, X_test)
    oof_preds[name]  = oof
    test_preds[name] = test_pred

# Score du blending simple (pondéré par inverse RMSLE)
stack_df = pd.DataFrame(oof_preds)
stack_test_df = pd.DataFrame(test_preds)

# Blending pondéré : plus le RMSLE est bas, plus le poids est élevé
model_rmsles = {name: rmsle(y.values, oof_preds[name]) for name in candidates}
inv_rmsle  = {k: 1/v for k, v in model_rmsles.items()}
total_inv  = sum(inv_rmsle.values())
weights    = {k: v/total_inv for k, v in inv_rmsle.items()}

print("\n  Scores OOF individuels :")
for name, score in sorted(model_rmsles.items(), key=lambda x: x[1]):
    print(f"    {name:<12} RMSLE = {score:.4f}  poids = {weights[name]:.3f}")

blend_oof  = sum(weights[k] * oof_preds[k]  for k in candidates)
blend_test = sum(weights[k] * test_preds[k] for k in candidates)
blend_rmsle = rmsle(y.values, blend_oof)
print(f"\n  Blending pondéré      RMSLE = {blend_rmsle:.4f}")

# Stacking avec méta-modèle Ridge
from sklearn.linear_model import Ridge as RidgeMeta
meta = RidgeMeta(alpha=1.0)
meta.fit(stack_df, y)
stack_oof   = meta.predict(stack_df)
stack_test_ = meta.predict(stack_test_df)
stack_rmsle = rmsle(y.values, stack_oof)
print(f"  Stacking (Ridge méta) RMSLE = {stack_rmsle:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6 : SÉLECTION DU MEILLEUR & SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 6 — Sélection et soumission")
print("=" * 60)

if blend_rmsle <= stack_rmsle:
    best_pred_log = blend_test
    best_name     = "Blending pondéré"
    best_rmsle    = blend_rmsle
else:
    best_pred_log = stack_test_
    best_name     = "Stacking Ridge"
    best_rmsle    = stack_rmsle

print(f"\n  Meilleure stratégie : {best_name}  (RMSLE CV = {best_rmsle:.4f})")

# Reconvertir
final_pred = np.maximum(np.expm1(best_pred_log), 100_000)

# Submission
submission = pd.DataFrame({
    'id':   test_fe['id'].values,
    'prix': final_pred.round(0).astype(int),
})
sub_path = os.path.join(OUT_DIR, 'submission_improved.csv')
submission.to_csv(sub_path, index=False)
print(f"  Soumission : {sub_path} ({len(submission)} lignes)")
print(f"  Prix min : {submission['prix'].min():,} MRU")
print(f"  Prix max : {submission['prix'].max():,} MRU")
print(f"  Prix médian : {submission['prix'].median():,.0f} MRU")

# ─────────────────────────────────────────────────────────────────────────────
# TABLEAU RÉCAPITULATIF
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TABLEAU COMPARATIF")
print("=" * 60)
print(f"\n  {'Modèle':<30} {'RMSLE CV':>10}")
print(f"  {'-'*42}")
print(f"  {'Baseline (médiane)':<30} {'~0.999':>10}")
print(f"  {'Ancien RF (34 features)':<30} {'0.6576':>10}")
for name, score in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {name:<30} {score:>10.4f}")
for name, score in [('Blending pondéré', blend_rmsle), ('Stacking Ridge', stack_rmsle)]:
    marker = " ← MEILLEUR" if name == best_name else ""
    print(f"  {name:<30} {score:>10.4f}{marker}")

print(f"\n  Amélioration vs ancien modèle : {0.6576 - best_rmsle:+.4f} RMSLE")

# Sauvegarder le meilleur modèle individuel
best_single = min(model_rmsles.items(), key=lambda x: x[1])
print(f"\n  Meilleur modèle individuel : {best_single[0]} (RMSLE={best_single[1]:.4f})")
best_single_model = candidates[best_single[0]]
best_single_model.fit(X_train, y)
joblib.dump(best_single_model, os.path.join(MODEL_DIR, 'housing_model_improved.pkl'))
joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, 'features_improved.pkl'))
print("  Modèle sauvegardé : model/housing_model_improved.pkl")

# Mettre à jour geo_meta.json avec le RMSLE CV réel (calculé, pas hardcodé)
with open(GEO_META, encoding='utf-8') as f:
    _geo_out = json.load(f)
_geo_out['cv_rmsle'] = round(best_rmsle, 6)
with open(GEO_META, 'w', encoding='utf-8') as f:
    json.dump(_geo_out, f, ensure_ascii=False, indent=2)
print(f"  geo_meta.json mis à jour avec cv_rmsle = {best_rmsle:.4f}")
