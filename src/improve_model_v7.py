#!/usr/bin/env python3
"""
improve_model_v7.py — Fix early stopping bug + property type + source
V7 Changes over V6b:
  1. EARLY STOPPING IN OOF LOOP: LGB/XGB/CatBoost were running all n_estimators
     in the stacking OOF loop (no early stopping), causing overfitting per fold.
     LGB OOF was 0.6032 vs Optuna 0.5775 — a 0.026 gap from this alone.
     Now uses eval_set on val fold with early_stopping in the OOF loop too.
  2. PROPERTY TYPE EXTRACTION: Extracts villa, appartement, terrain, immeuble,
     duplex, studio from titre+description. Fundamental price signal — a 200m²
     terrain vs villa in the same quartier have completely different prices.
  3. SOURCE ENCODING: frequency-encode the 'source' column (5 scraped sites).
     Agency listings (lagence-mr) vs private ads have different price profiles.
  4. FIX Ridge+feats meta: was blowing up at 0.9081 due to numerical instability.
     Replaced with a StandardScaler before Ridge and capped N_META_FEATS at 5.
  5. LOO TARGET ENCODING: re-added from V6 (was dropped in V6b revert).
"""

import sys, os, re, warnings, time
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

OPTUNA_TRIALS = 100       # trials per model
N_FOLDS       = 5
RANDOM_STATE  = 42
SVD_COMPONENTS_PER_LANG = 25  # 25 FR + 25 AR = 50 total (transductive fit allows more)
PSEUDO_ROUNDS  = 2            # number of pseudo-labeling iterations
PSEUDO_WEIGHT  = 0.35         # weight for pseudo-labeled samples vs real labels (0–1)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

with open(GEO_META, encoding='utf-8') as f:
    _geo = json.load(f)

QUARTIER_META = _geo['quartiers']
QUARTIER_GPS  = {q: (v['lat'], v['lon']) for q, v in QUARTIER_META.items()}
QUARTIER_POI  = {
    q: (v['poi']['ecoles'], v['poi']['mosquees'], v['poi']['commerces'], v['poi']['hopitaux'])
    for q, v in QUARTIER_META.items()
}
_ref         = _geo['reference_points']
CENTRE_GPS   = (_ref['centre']['lat'],   _ref['centre']['lon'])
AEROPORT_GPS = (_ref['aeroport']['lat'], _ref['aeroport']['lon'])
PLAGE_GPS    = (_ref['plage']['lat'],    _ref['plage']['lon'])


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def rmsle(y_true_log, y_pred_log):
    y_true = np.expm1(np.clip(y_true_log, 0, None))
    y_pred = np.expm1(np.clip(y_pred_log, 0, None))
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def haversine(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def arabic_ratio(text):
    """Fraction of characters that are Arabic (U+0600–U+06FF)."""
    if not text or len(text) == 0:
        return 0.0
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return arabic_chars / len(text)


def extract_price_from_text(text):
    """
    Extract explicit price (MRU) from listing text. Conservative: only fires
    when there is an explicit label or unit (MRU, million, prix:).
    NO bare number matching — that catches phone numbers in Mauritania.
    Returns float MRU if plausible (100k–600M), else np.nan.
    """
    if not text:
        return np.nan
    t = str(text).lower().strip()

    # Pattern 1: explicit MRU/ouguiya mention
    m = re.search(r'([\d][\d\s]{3,}[\d])\s*(?:mru|ouguiya|أوقية|ougiya)', t)
    if m:
        try:
            val = float(re.sub(r'\s', '', m.group(1)))
            if 100_000 <= val <= 600_000_000:
                return val
        except Exception:
            pass

    # Pattern 2: "X millions" or "X,Y millions"
    m = re.search(r'([\d]+[.,]?[\d]*)\s*(?:millions?|مليون)', t)
    if m:
        try:
            val = float(m.group(1).replace(',', '.')) * 1_000_000
            if 100_000 <= val <= 600_000_000:
                return val
        except Exception:
            pass

    # Pattern 3: price-labelled number "prix: 5000000" or "prix 5 000 000"
    m = re.search(r'(?:prix|price|سعر)[:\s]+([\d][\d\s]{4,}[\d])', t)
    if m:
        try:
            val = float(re.sub(r'\s', '', m.group(1)))
            if 100_000 <= val <= 600_000_000:
                return val
        except Exception:
            pass

    return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Load & Clean")
print("=" * 60)

train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

for df in [train, test]:
    df['quartier'] = df['quartier'].str.strip()
    df['date_publication'] = pd.to_datetime(df['date_publication'], errors='coerce')
    df['caracteristiques'] = df['caracteristiques'].fillna('')
    df['description']      = df['description'].fillna('')
    df['titre']            = df['titre'].fillna('')

print(f"Train: {train.shape} | Test: {test.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: IMPUTATION (fit on train only)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Imputation")
print("=" * 60)

for df in [train, test]:
    df['has_sdb_info'] = df['nb_sdb'].notna().astype(int)

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

# Surface sanity: clip to realistic range for Nouakchott residential listings
for df in [train, test]:
    df['surface_m2'] = df['surface_m2'].clip(lower=20, upper=2000)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING (NO target encoding — that goes inside CV)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Feature Engineering")
print("=" * 60)


def build_features(train_df, test_df):
    """
    Build all features EXCEPT target encoding (which must be done inside CV folds).
    Everything that needs a fit (TF-IDF, medians) is fit on train_df only.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    dfs = {'train': train_df.copy(), 'test': test_df.copy()}

    # ── A) Combined text ─────────────────────────────────────────────────
    for name, df in dfs.items():
        df['text_combined'] = (
            df['titre'] + ' ' + df['description'] + ' ' + df['caracteristiques']
        ).str.lower()

    # ── B) Dual-language TF-IDF + SVD ────────────────────────────────────
    print("  Dual-language TF-IDF + SVD (transductive: fit on train+test)...")

    # Detect language per listing
    for name, df in dfs.items():
        df['_ar_ratio'] = df['text_combined'].apply(arabic_ratio)
        df['is_arabic_listing'] = (df['_ar_ratio'] > 0.3).astype(int)

    # Transductive NLP: fit vocabulary on ALL available text (train+test).
    # No label info is used — IDF weights from test texts are legitimate.
    all_text_combined = pd.concat(
        [dfs['train']['text_combined'], dfs['test']['text_combined']], ignore_index=True
    )

    # French pipeline
    tfidf_fr = TfidfVectorizer(max_features=300, min_df=2, max_df=0.95, ngram_range=(1, 2))
    tfidf_fr.fit(all_text_combined)
    tfidf_fr_train = tfidf_fr.transform(dfs['train']['text_combined'])
    tfidf_fr_test  = tfidf_fr.transform(dfs['test']['text_combined'])

    svd_fr = TruncatedSVD(n_components=SVD_COMPONENTS_PER_LANG, random_state=RANDOM_STATE)
    # Fit SVD on all text to capture full variance
    svd_fr.fit(tfidf_fr.transform(all_text_combined))
    svd_fr_train = svd_fr.transform(tfidf_fr_train)
    svd_fr_test  = svd_fr.transform(tfidf_fr_test)

    # Arabic pipeline — char n-grams work better for Arabic
    tfidf_ar = TfidfVectorizer(
        max_features=200, min_df=2, max_df=0.95,
        ngram_range=(1, 2), analyzer='char_wb',
        token_pattern=r'[\u0600-\u06FF]+'
    )
    # Build Arabic-only text (keep only Arabic chars + spaces)
    ar_text_train = dfs['train']['text_combined'].apply(
        lambda t: ' '.join(re.findall(r'[\u0600-\u06FF]+', t)))
    ar_text_test = dfs['test']['text_combined'].apply(
        lambda t: ' '.join(re.findall(r'[\u0600-\u06FF]+', t)))
    all_ar_text = pd.concat([ar_text_train, ar_text_test], ignore_index=True)

    tfidf_ar.fit(all_ar_text)
    tfidf_ar_train = tfidf_ar.transform(ar_text_train)
    tfidf_ar_test  = tfidf_ar.transform(ar_text_test)

    svd_ar = TruncatedSVD(n_components=SVD_COMPONENTS_PER_LANG, random_state=RANDOM_STATE)
    svd_ar.fit(tfidf_ar.transform(all_ar_text))
    svd_ar_train = svd_ar.transform(tfidf_ar_train)
    svd_ar_test  = svd_ar.transform(tfidf_ar_test)

    # Store SVD components
    for i in range(SVD_COMPONENTS_PER_LANG):
        dfs['train'][f'tfidf_fr_{i}'] = svd_fr_train[:, i]
        dfs['test'][f'tfidf_fr_{i}']  = svd_fr_test[:, i]
        dfs['train'][f'tfidf_ar_{i}'] = svd_ar_train[:, i]
        dfs['test'][f'tfidf_ar_{i}']  = svd_ar_test[:, i]

    # ── C) Keywords (FR + AR) ────────────────────────────────────────────
    print("  Keywords...")
    keywords = {
        'has_garage':        r'garage|كراج',
        'has_titre_foncier': r'titre foncier|وثيقة ملكية',
        'has_camera':        r'cam|كاميرا',
        'has_piscine':       r'piscine|مسبح',
        'has_clim':          r'climatisation|clim|مكيف',
        'has_meuble':        r'meubl|مفروش',
        'is_luxe':           r'luxe|haut standing|standing|فاخر',
        'is_renove':         r'rénov|renov|neuf|nouveau|جديد|مجدد',
        'has_etage':         r'étage|etage|طابق|طوابق',
        'has_duplex':        r'duplex|دوبلكس',
        'has_jardin':        r'jardin|حديقة',
        'has_terrasse':      r'terrasse|تراس|سطح',
        'has_cuisine_equip': r'cuisine.quip|مطبخ مجهز',
        'has_carrelage':     r'carrelage|كارو|caro',
        'has_2_facades':     r'façade|واجه|facade',
        'mention_coin':      r'\bcoin\b|ركن|angle',
        'mention_prix_neg':  r'négociable|قابل للتفاوض|negotiable',
        'mention_urgent':    r'\burgent\b|مستعجل|فرصة',
        # has_arabic computed below from _ar_ratio (avoid PyArrow \u regex issue)
        'desc_has_numbers':  r'\d{6,}',
    }

    for name, df in dfs.items():
        combined = df['text_combined']
        for feat, pattern in keywords.items():
            df[feat] = combined.str.contains(pattern, na=False, regex=True).astype(int)
        df['desc_len']        = df['description'].str.len().fillna(0)
        df['desc_word_count'] = df['description'].str.split().str.len().fillna(0)
        df['desc_nb_digits']  = df['description'].str.count(r'\d').fillna(0)
        # Derived from _ar_ratio (avoids PyArrow \u regex limitation in str.contains)
        df['has_arabic'] = (df['_ar_ratio'] > 0).astype(int)

    # ── D) Structured text extraction (NEW in V3) ────────────────────────
    print("  Structured text extraction...")
    for name, df in dfs.items():
        text = df['text_combined']
        # Phone numbers (signal of seriousness)
        df['has_phone'] = text.str.contains(r'\d{8,}', na=False, regex=True).astype(int)
        # Explicit price mentions in text
        df['has_price_mention'] = text.str.contains(
            r'\d+[\s.]*(?:mru|million|مليون)', na=False, regex=True).astype(int)
        # Safe price extraction — explicit units/labels only (no bare numbers)
        df['desc_price_extracted'] = df['text_combined'].apply(extract_price_from_text)
        df['has_extracted_price']  = df['desc_price_extracted'].notna().astype(int)
        df['desc_price_log']       = np.log1p(df['desc_price_extracted'].fillna(0))
        # Floor count extraction
        floor_extract = text.str.extract(r'(\d+)\s*(?:étage|etage|طابق)', expand=False)
        df['n_floors'] = pd.to_numeric(floor_extract, errors='coerce').fillna(0)
        # Year of construction
        year_extract = text.str.extract(r'(20[0-2]\d)', expand=False)
        df['construction_year'] = pd.to_numeric(year_extract, errors='coerce').fillna(0)
        # Description length buckets (quantile-based, fit on train)
    # Fit quantile bins on train
    desc_q = dfs['train']['desc_len'].quantile([0.33, 0.66])
    for name, df in dfs.items():
        df['desc_len_bucket'] = np.where(
            df['desc_len'] <= desc_q.iloc[0], 0,
            np.where(df['desc_len'] <= desc_q.iloc[1], 1, 2))

    # ── E) Extraction from caracteristiques ──────────────────────────────
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

    # ── F) Geo features ──────────────────────────────────────────────────
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
        df['nb_total_pois_1km'] = (
            df['nb_ecoles_1km'] + df['nb_mosquees_1km'] +
            df['nb_commerces_1km'] + df['nb_hopitaux_1km']
        )

    # ── G) Derived structural features ───────────────────────────────────
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

    # ── H) Quartier frequency (leak-free — no target info) ──────────────
    q_freq = dfs['train']['quartier'].value_counts(normalize=True)
    for name, df in dfs.items():
        df['quartier_freq'] = df['quartier'].map(q_freq).fillna(1 / 8)

    # ── I) Property type extraction (V7 — high signal) ───────────────────
    print("  Property type extraction...")
    type_patterns = {
        'is_villa':        r'villa|فيلا',
        'is_appartement':  r'appartement|appart\b|شقة',
        'is_terrain':      r'terrain\b|terrain nu|ارض|قطعة',
        'is_immeuble':     r'immeuble|عمارة|بناية',
        'is_duplex':       r'duplex|دوبلكس',
        'is_studio':       r'studio|ستوديو',
    }
    for name, df in dfs.items():
        text = df['text_combined']
        for feat, pat in type_patterns.items():
            df[feat] = text.str.contains(pat, na=False, regex=True).astype(int)
        # nb_types_mentioned as a sanity signal (0 = unknown type)
        df['nb_type_mentions'] = sum(df[f] for f in type_patterns)

    # ── J) Source encoding (frequency, fit on train) ──────────────────────
    src_freq = dfs['train']['source'].value_counts(normalize=True)
    for name, df in dfs.items():
        df['source_freq'] = df['source'].map(src_freq).fillna(src_freq.min())

    return dfs['train'], dfs['test']


train_fe, test_fe = build_features(train, test)


# ── Global outlier clipping (conservative) ───────────────────────────────────
# V5: replaced per-quartier P2-P98 with global P0.5-P99.5.
# Rationale: per-quartier clipping on small quartiers (13–30 listings) produces
# unreliable quantile estimates and was capping legitimate high-value properties
# in Tevragh Zeina, causing systematic under-prediction on the expensive end.
print("\n  Global outlier clipping (P0.5–P99.5)...")
lo = train_fe['prix'].quantile(0.005)
hi = train_fe['prix'].quantile(0.995)
n_clipped = ((train_fe['prix'] < lo) | (train_fe['prix'] > hi)).sum()
train_fe['prix'] = train_fe['prix'].clip(lo, hi)
print(f"    Clipped {n_clipped} rows to [{lo:,.0f}, {hi:,.0f}] MRU")

y = np.log1p(train_fe['prix'])

# ── Feature columns ──────────────────────────────────────────────────────────
TFIDF_FR_COLS = [f'tfidf_fr_{i}' for i in range(SVD_COMPONENTS_PER_LANG)]
TFIDF_AR_COLS = [f'tfidf_ar_{i}' for i in range(SVD_COMPONENTS_PER_LANG)]

# NOTE: target encoding cols are NOT here — they're computed inside CV folds
BASE_FEATURES = [
    'surface_m2', 'nb_chambres', 'nb_salons', 'nb_sdb', 'has_sdb_info',
    'latitude', 'longitude', 'dist_centre_km', 'dist_aeroport_km', 'dist_plage_km',
    'nb_ecoles_1km', 'nb_mosquees_1km', 'nb_commerces_1km', 'nb_hopitaux_1km', 'nb_total_pois_1km',
    'nb_pieces_total', 'has_garage', 'has_titre_foncier', 'has_camera', 'nb_balcons', 'taille_rue',
    'desc_len', 'desc_word_count', 'desc_has_numbers', 'desc_nb_digits',
    'has_piscine', 'has_clim', 'has_meuble', 'is_luxe', 'is_renove', 'has_arabic',
    'has_etage', 'has_duplex', 'has_jardin', 'has_terrasse', 'has_carrelage',
    'has_cuisine_equip',  # ← was missing in V2!
    'mention_coin', 'mention_prix_neg', 'mention_urgent', 'has_2_facades',
    'age_annonce_jours', 'mois_publication', 'trimestre', 'is_weekend',
    'surface_par_piece', 'log_surface', 'log_chambres', 'surface_squared', 'densite_pieces',
    'quartier_freq',
    # NEW in V3
    'is_arabic_listing', 'has_phone', 'has_price_mention', 'n_floors',
    'construction_year', 'desc_len_bucket',
    # NEW in V6b — conservative price extraction
    'desc_price_extracted', 'has_extracted_price', 'desc_price_log',
    # NEW in V7 — property type + source
    'is_villa', 'is_appartement', 'is_terrain', 'is_immeuble', 'is_duplex',
    'is_studio', 'nb_type_mentions', 'source_freq',
] + TFIDF_FR_COLS + TFIDF_AR_COLS

# Target-encoding columns added inside CV folds:
TARGET_ENC_COLS = [
    'quartier_target_enc', 'quartier_target_smooth', 'quartier_prix_std',
    'surface_x_target', 'chambres_vs_quartier', 'surface_vs_quartier',
    'log_surface_x_target',
]

ALL_FEATURE_COLS = BASE_FEATURES + TARGET_ENC_COLS

# Validate that base features exist
FEATURE_COLS = [c for c in BASE_FEATURES if c in train_fe.columns]
missing = set(BASE_FEATURES) - set(FEATURE_COLS)
if missing:
    print(f"  WARNING: missing base features: {missing}")
print(f"\n  Base features: {len(FEATURE_COLS)} | + target enc: {len(TARGET_ENC_COLS)} = {len(ALL_FEATURE_COLS)} total")

# Fill NaN in base features
feature_medians = train_fe[FEATURE_COLS].median()
train_fe[FEATURE_COLS] = train_fe[FEATURE_COLS].fillna(feature_medians)
test_fe[FEATURE_COLS]  = test_fe[FEATURE_COLS].fillna(feature_medians)


# ─────────────────────────────────────────────────────────────────────────────
# TARGET ENCODING HELPER (applied per-fold to avoid leakage)
# ─────────────────────────────────────────────────────────────────────────────
def apply_target_encoding(df, quartier_col, target_series, global_mean, smoothing=30):
    """
    Leave-One-Out (LOO) target encoding — fit on training fold only.
    Train rows: encoding excludes the row itself (true LOO, no leakage).
    Val/test rows: use full quartier mean from train fold.
    Caches stats as function attributes for val/test transform.
    """
    if len(target_series) == len(df):
        # Fit mode on training fold
        quartiers = df[quartier_col].values
        targets   = target_series.values

        q_sum, q_count = {}, {}
        for q, t in zip(quartiers, targets):
            q_sum[q]   = q_sum.get(q, 0.0) + t
            q_count[q] = q_count.get(q, 0) + 1

        # LOO per row
        loo_enc = np.array([
            (q_sum[q] - t) / (q_count[q] - 1) if q_count[q] > 1 else global_mean
            for q, t in zip(quartiers, targets)
        ])

        q_mean = {q: q_sum[q] / q_count[q] for q in q_sum}
        q_sq   = {}
        for q, t in zip(quartiers, targets):
            q_sq[q] = q_sq.get(q, 0.0) + (t - q_mean[q]) ** 2
        q_std = {q: np.sqrt(q_sq[q] / q_count[q]) for q in q_sq}

        q_ch_med  = df.groupby(quartier_col)['nb_chambres'].median()
        q_sur_med = df.groupby(quartier_col)['surface_m2'].median()

        apply_target_encoding._q_mean    = q_mean
        apply_target_encoding._q_std     = q_std
        apply_target_encoding._q_ch_med  = q_ch_med
        apply_target_encoding._q_sur_med = q_sur_med
        apply_target_encoding._global    = global_mean

        df = df.copy()
        df['quartier_target_enc']    = loo_enc
        df['quartier_target_smooth'] = loo_enc
        df['quartier_prix_std']      = df[quartier_col].map(q_std).fillna(np.nanmean(list(q_std.values())))
        df['surface_x_target']       = df['surface_m2'] * loo_enc / 1e6
        df['chambres_vs_quartier']   = df['nb_chambres'] - df[quartier_col].map(q_ch_med).fillna(3)
        df['surface_vs_quartier']    = df['surface_m2']  - df[quartier_col].map(q_sur_med).fillna(200)
        df['log_surface_x_target']   = df['log_surface'] * loo_enc / 1e6
    else:
        # Transform mode: val/test uses cached train-fold stats
        q_mean    = apply_target_encoding._q_mean
        q_std     = apply_target_encoding._q_std
        q_ch_med  = apply_target_encoding._q_ch_med
        q_sur_med = apply_target_encoding._q_sur_med
        global_mean = apply_target_encoding._global

        df = df.copy()
        enc = df[quartier_col].map(q_mean).fillna(global_mean).values
        df['quartier_target_enc']    = enc
        df['quartier_target_smooth'] = enc
        df['quartier_prix_std']      = df[quartier_col].map(q_std).fillna(np.nanmean(list(q_std.values())))
        df['surface_x_target']       = df['surface_m2'] * enc / 1e6
        df['chambres_vs_quartier']   = df['nb_chambres'] - df[quartier_col].map(q_ch_med).fillna(3)
        df['surface_vs_quartier']    = df['surface_m2']  - df[quartier_col].map(q_sur_med).fillna(200)
        df['log_surface_x_target']   = df['log_surface'] * enc / 1e6
    return df


def prepare_fold_data(train_fe, y, train_idx, val_idx):
    """Prepare train/val data with per-fold LOO target encoding."""
    X_tr  = train_fe.iloc[train_idx].copy().reset_index(drop=True)
    X_val = train_fe.iloc[val_idx].copy().reset_index(drop=True)

    prix_train_fold = np.expm1(y.iloc[train_idx].values)
    global_mean     = prix_train_fold.mean()
    y_tr_series     = pd.Series(prix_train_fold, index=X_tr.index)

    X_tr  = apply_target_encoding(X_tr,  'quartier', y_tr_series, global_mean)
    X_val = apply_target_encoding(X_val, 'quartier', y_tr_series, global_mean)

    return X_tr[ALL_FEATURE_COLS].fillna(0), X_val[ALL_FEATURE_COLS].fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: OPTUNA HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Optuna Hyperparameter Tuning")
print("=" * 60)

import optuna
from sklearn.model_selection import KFold
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Two separate KFold objects:
# kf_tune   → used ONLY during Optuna trials (different seed → prevents fold-level overfitting)
# kf        → used for final CV evaluation and stacking OOF generation
kf_tune = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + 57)
kf      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ── 4a) Tune LightGBM ───────────────────────────────────────────────────────
print("\n  Tuning LightGBM...")
import lightgbm as lgb

def lgb_objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate':    trial.suggest_float('lr', 0.01, 0.1, log=True),
        'max_depth':        trial.suggest_int('depth', 4, 10),
        'num_leaves':       trial.suggest_int('num_leaves', 31, 127),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample', 0.5, 1.0),
        'reg_alpha':        trial.suggest_float('alpha', 1e-3, 10, log=True),
        'reg_lambda':       trial.suggest_float('lambda', 1e-3, 10, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1,
    }
    scores = []
    for train_idx, val_idx in kf_tune.split(train_fe):
        X_tr, X_val = prepare_fold_data(train_fe, y, train_idx, val_idx)
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        pred = model.predict(X_val)
        scores.append(rmsle(y_val.values, pred))
    return np.mean(scores)

t0 = time.time()
lgb_study = optuna.create_study(direction='minimize')
lgb_study.optimize(lgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  LightGBM best RMSLE: {lgb_study.best_value:.4f} ({time.time()-t0:.0f}s)")
print(f"  Best params: {lgb_study.best_params}")

# ── 4b) Tune CatBoost ───────────────────────────────────────────────────────
print("\n  Tuning CatBoost...")
from catboost import CatBoostRegressor

def cat_objective(trial):
    params = {
        'iterations':        trial.suggest_int('iterations', 500, 3000),
        'learning_rate':     trial.suggest_float('lr', 0.01, 0.1, log=True),
        'depth':             trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg':       trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample', 0.5, 1.0),
        'min_data_in_leaf':  trial.suggest_int('min_data_in_leaf', 1, 30),
        'random_seed': RANDOM_STATE, 'verbose': 0,
        'early_stopping_rounds': 50,
    }
    scores = []
    for train_idx, val_idx in kf_tune.split(train_fe):
        X_tr, X_val = prepare_fold_data(train_fe, y, train_idx, val_idx)
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        pred = model.predict(X_val)
        scores.append(rmsle(y_val.values, pred))
    return np.mean(scores)

t0 = time.time()
cat_study = optuna.create_study(direction='minimize')
cat_study.optimize(cat_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  CatBoost best RMSLE: {cat_study.best_value:.4f} ({time.time()-t0:.0f}s)")
print(f"  Best params: {cat_study.best_params}")

# ── 4c) Tune XGBoost ────────────────────────────────────────────────────────
print("\n  Tuning XGBoost...")
import xgboost as xgb_mod  # alias to avoid conflict with later import

def xgb_objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate':    trial.suggest_float('lr', 0.01, 0.1, log=True),
        'max_depth':        trial.suggest_int('depth', 3, 9),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample', 0.5, 1.0),
        'reg_alpha':        trial.suggest_float('alpha', 1e-3, 10, log=True),
        'reg_lambda':       trial.suggest_float('lambda', 1e-3, 10, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma':            trial.suggest_float('gamma', 0, 1.0),
        'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0,
        'early_stopping_rounds': 50, 'eval_metric': 'rmse',
    }
    scores = []
    for train_idx, val_idx in kf_tune.split(train_fe):
        X_tr, X_val = prepare_fold_data(train_fe, y, train_idx, val_idx)
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = xgb_mod.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        scores.append(rmsle(y_val.values, model.predict(X_val)))
    return np.mean(scores)

t0 = time.time()
xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  XGBoost best RMSLE: {xgb_study.best_value:.4f} ({time.time()-t0:.0f}s)")
print(f"  Best params: {xgb_study.best_params}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: CROSS-VALIDATION WITH TUNED MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Model Comparison (CV with leak-free target encoding)")
print("=" * 60)

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
)
import xgboost as xgb
import joblib

results = {}


def cv_score_leak_free(model_fn, X_full, y, name, use_cat_features=False):
    """CV with per-fold target encoding. model_fn returns a fresh model instance."""
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_full)):
        X_tr, X_val = prepare_fold_data(X_full, y, tr_idx, val_idx)
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = model_fn()

        if use_cat_features:
            # CatBoost with native categoricals: add quartier as raw string
            X_tr_cat = X_tr.copy()
            X_val_cat = X_val.copy()
            X_tr_cat['quartier'] = X_full.iloc[tr_idx]['quartier'].values
            X_val_cat['quartier'] = X_full.iloc[val_idx]['quartier'].values
            cat_idx = [X_tr_cat.columns.get_loc('quartier')]
            model.fit(X_tr_cat, y_tr, eval_set=(X_val_cat, y_val), verbose=0)
            pred = model.predict(X_val_cat)
        else:
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)

        scores.append(rmsle(y_val.values, pred))
    mean_rmsle = np.mean(scores)
    std_rmsle  = np.std(scores)
    print(f"  {name:<30} RMSLE = {mean_rmsle:.4f} ± {std_rmsle:.4f}")
    results[name] = mean_rmsle
    return mean_rmsle


# Build tuned model factories
lgb_best_p = lgb_study.best_params.copy()
lgb_params = {
    'n_estimators':      lgb_best_p.get('n_estimators', 1000),
    'learning_rate':     lgb_best_p.get('lr', 0.03),
    'max_depth':         lgb_best_p.get('depth', 6),
    'num_leaves':        lgb_best_p.get('num_leaves', 63),
    'subsample':         lgb_best_p.get('subsample', 0.8),
    'colsample_bytree':  lgb_best_p.get('colsample', 0.7),
    'reg_alpha':         lgb_best_p.get('alpha', 0.1),
    'reg_lambda':        lgb_best_p.get('lambda', 1.0),
    'min_child_samples': lgb_best_p.get('min_child_samples', 10),
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1,
}

cat_best_p = cat_study.best_params.copy()
cat_params = {
    'iterations':        cat_best_p.get('iterations', 1000),
    'learning_rate':     cat_best_p.get('lr', 0.03),
    'depth':             cat_best_p.get('depth', 6),
    'l2_leaf_reg':       cat_best_p.get('l2_leaf_reg', 3),
    'subsample':         cat_best_p.get('subsample', 0.8),
    'colsample_bylevel': cat_best_p.get('colsample', 0.7),
    'min_data_in_leaf':  cat_best_p.get('min_data_in_leaf', 5),
    'random_seed': RANDOM_STATE, 'verbose': 0,
}

cat_native_params = cat_params.copy()
cat_native_params['cat_features'] = ['quartier']

xgb_best_p = xgb_study.best_params.copy()
xgb_params = {
    'n_estimators':     xgb_best_p.get('n_estimators', 1000),
    'learning_rate':    xgb_best_p.get('lr', 0.03),
    'max_depth':        xgb_best_p.get('depth', 6),
    'subsample':        xgb_best_p.get('subsample', 0.8),
    'colsample_bytree': xgb_best_p.get('colsample', 0.7),
    'reg_alpha':        xgb_best_p.get('alpha', 0.1),
    'reg_lambda':       xgb_best_p.get('lambda', 1.0),
    'min_child_weight': xgb_best_p.get('min_child_weight', 5),
    'gamma':            xgb_best_p.get('gamma', 0.1),
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0,
}

cv_score_leak_free(lambda: Ridge(alpha=1.0), train_fe, y, "Ridge")
cv_score_leak_free(
    lambda: RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3,
                                   max_features=0.6, random_state=RANDOM_STATE, n_jobs=-1),
    train_fe, y, "Random Forest")
cv_score_leak_free(
    lambda: ExtraTreesRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3,
                                 max_features=0.6, random_state=RANDOM_STATE, n_jobs=-1),
    train_fe, y, "ExtraTrees")
cv_score_leak_free(
    lambda: xgb.XGBRegressor(**xgb_params),
    train_fe, y, "XGBoost (tuned)")
cv_score_leak_free(
    lambda: lgb.LGBMRegressor(**lgb_params),
    train_fe, y, "LightGBM (tuned)")
cv_score_leak_free(
    lambda: CatBoostRegressor(**cat_params),
    train_fe, y, "CatBoost (tuned)")
cv_score_leak_free(
    lambda: CatBoostRegressor(**{k: v for k, v in cat_native_params.items() if k != 'cat_features'},
                               cat_features=['quartier']),
    train_fe, y, "CatBoost (native cat)", use_cat_features=True)
cv_score_leak_free(
    lambda: GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                       min_samples_leaf=5, subsample=0.8, random_state=RANDOM_STATE),
    train_fe, y, "GradientBoosting")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: STACKING / BLENDING (with leak-free OOF + raw features in meta)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Enhanced Stacking")
print("=" * 60)

base_model_factories = {
    'lgb':   lambda: lgb.LGBMRegressor(**lgb_params),
    'cat':   lambda: CatBoostRegressor(**cat_params),
    'xgb':   lambda: xgb.XGBRegressor(**xgb_params),
    'rf':    lambda: RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_leaf=3,
        max_features=0.6, random_state=RANDOM_STATE, n_jobs=-1),
    'et':    lambda: ExtraTreesRegressor(
        n_estimators=300, max_depth=15, min_samples_leaf=3,
        max_features=0.6, random_state=RANDOM_STATE, n_jobs=-1),
    'ridge': lambda: Ridge(alpha=1.0),
}

oof_preds  = {}
test_preds = {}

# Apply target encoding on full train for test predictions
prix_full = np.expm1(y)
global_mean_full = prix_full.mean()
train_full_enc = apply_target_encoding(train_fe, 'quartier', prix_full, global_mean_full)
test_full_enc  = apply_target_encoding(test_fe, 'quartier', prix_full, global_mean_full)

X_train_full = train_full_enc[ALL_FEATURE_COLS].fillna(0)
X_test_full  = test_full_enc[ALL_FEATURE_COLS].fillna(0)

for name, model_fn in base_model_factories.items():
    print(f"  OOF {name}...")
    oof_train = np.zeros(len(train_fe))
    oof_test_folds = np.zeros((len(test_fe), N_FOLDS))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_fe)):
        X_tr, X_val = prepare_fold_data(train_fe, y, tr_idx, val_idx)
        y_tr = y.iloc[tr_idx]
        y_val_fold = y.iloc[val_idx]
        model = model_fn()

        # Use early stopping for boosting models to avoid over-running n_estimators
        if name == 'lgb':
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val_fold)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(0)])
        elif name == 'xgb':
            model.set_params(early_stopping_rounds=50, eval_metric='rmse')
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val_fold)],
                      verbose=False)
        elif name == 'cat':
            model.set_params(early_stopping_rounds=50)
            model.fit(X_tr, y_tr,
                      eval_set=(X_val, y_val_fold),
                      verbose=0)
        else:
            model.fit(X_tr, y_tr)

        oof_train[val_idx]       = model.predict(X_val)
        oof_test_folds[:, fold]  = model.predict(X_test_full)

    oof_preds[name]  = oof_train
    test_preds[name] = oof_test_folds.mean(axis=1)

# Individual OOF scores
model_rmsles = {name: rmsle(y.values, oof_preds[name]) for name in base_model_factories}
print("\n  OOF scores:")
for name, score in sorted(model_rmsles.items(), key=lambda x: x[1]):
    print(f"    {name:<12} RMSLE = {score:.4f}")

# ── Weighted blending ────────────────────────────────────────────────────────
inv_rmsle = {k: 1 / v for k, v in model_rmsles.items()}
total_inv = sum(inv_rmsle.values())
weights   = {k: v / total_inv for k, v in inv_rmsle.items()}

blend_oof  = sum(weights[k] * oof_preds[k] for k in base_model_factories)
blend_test = sum(weights[k] * test_preds[k] for k in base_model_factories)
blend_rmsle_val = rmsle(y.values, blend_oof)
print(f"\n  Blending (weighted)       RMSLE = {blend_rmsle_val:.4f}")

# ── Stacking with Ridge meta — PROPER nested OOF evaluation ─────────────────
# BUGFIX V4: the meta-learner must be evaluated via its own CV loop, NOT by
# fitting on all OOF rows and predicting in-sample (which was the V3 bug).
#
# Strategy:
#   - For the CV score  → inner 5-fold CV on the OOF matrix
#   - For test preds    → fit on ALL OOF preds, predict on test OOF means
# This gives a true out-of-fold estimate for the meta-layer.

stack_df      = pd.DataFrame(oof_preds)
stack_test_df = pd.DataFrame(test_preds)

# ── Plain Ridge meta (no raw features) ───────────────────────────────────────
meta_oof_ridge  = np.zeros(len(y))
meta_test_ridge_folds = np.zeros((len(test_fe), N_FOLDS))

kf_meta = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + 13)
for fold, (tr_idx, val_idx) in enumerate(kf_meta.split(stack_df)):
    m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    m.fit(stack_df.iloc[tr_idx], y.values[tr_idx])
    meta_oof_ridge[val_idx]           = m.predict(stack_df.iloc[val_idx])
    meta_test_ridge_folds[:, fold]    = m.predict(stack_test_df)

stack_test_ridge  = meta_test_ridge_folds.mean(axis=1)
stack_rmsle_ridge = rmsle(y.values, meta_oof_ridge)
print(f"  Stacking (Ridge meta)     RMSLE = {stack_rmsle_ridge:.4f}")

# ── Ridge meta + top raw features — scaled, N_META_FEATS capped at 5 ─────────
# V7 fix: previous version blew up to 0.9081 due to correlated raw features
# causing numerical instability. Now: StandardScaler on meta inputs + 5 feats max.
from sklearn.preprocessing import StandardScaler as _SS
meta_oof_ridge2      = np.zeros(len(y))
meta_test_ridge2_folds = np.zeros((len(test_fe), N_FOLDS))

N_META_FEATS = 5  # reduced from 10 — fewer, more stable
for fold, (tr_idx, val_idx) in enumerate(kf_meta.split(stack_df)):
    corr_fold = (
        pd.DataFrame(X_train_full.values[tr_idx], columns=X_train_full.columns)
        .corrwith(pd.Series(y.values[tr_idx]))
        .abs()
        .sort_values(ascending=False)
    )
    # Exclude any target-encoding features (contain 'target' or 'prix') to be safe
    safe_feats = [f for f in corr_fold.index
                  if 'target' not in f and 'prix' not in f]
    top_feats = safe_feats[:N_META_FEATS]

    meta_tr_raw  = np.column_stack([stack_df.values[tr_idx],  X_train_full[top_feats].values[tr_idx]])
    meta_val_raw = np.column_stack([stack_df.values[val_idx], X_train_full[top_feats].values[val_idx]])
    meta_te_raw  = np.column_stack([stack_test_df.values,     X_test_full[top_feats].values])

    scaler = _SS()
    meta_tr  = scaler.fit_transform(meta_tr_raw)
    meta_val = scaler.transform(meta_val_raw)
    meta_te  = scaler.transform(meta_te_raw)

    m2 = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    m2.fit(meta_tr, y.values[tr_idx])
    meta_oof_ridge2[val_idx]           = m2.predict(meta_val)
    meta_test_ridge2_folds[:, fold]    = m2.predict(meta_te)

stack_test_ridge2  = meta_test_ridge2_folds.mean(axis=1)
stack_rmsle_ridge2 = rmsle(y.values, meta_oof_ridge2)
print(f"  Stacking (Ridge + feats)  RMSLE = {stack_rmsle_ridge2:.4f}")

# LightGBM meta-learner (non-linear) — uses kf_meta, inputs are OOF preds only
meta_oof_lgb  = np.zeros(len(y))
meta_test_lgb_folds = np.zeros((len(test_fe), N_FOLDS))
for fold, (tr_idx, val_idx) in enumerate(kf_meta.split(stack_df)):
    meta_lgb_fold = lgb.LGBMRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    meta_lgb_fold.fit(stack_df.values[tr_idx], y.values[tr_idx])
    meta_oof_lgb[val_idx]           = meta_lgb_fold.predict(stack_df.values[val_idx])
    meta_test_lgb_folds[:, fold]    = meta_lgb_fold.predict(stack_test_df.values)

meta_test_lgb = meta_test_lgb_folds.mean(axis=1)
stack_rmsle_lgb = rmsle(y.values, meta_oof_lgb)
print(f"  Stacking (LGB meta+feats) RMSLE = {stack_rmsle_lgb:.4f}")

# Pick the best base ensemble for use as pseudo-label generator
strategies_initial = {
    'Blending':          (blend_rmsle_val,    blend_test),
    'Stack Ridge':       (stack_rmsle_ridge,  stack_test_ridge),
    'Stack Ridge+feats': (stack_rmsle_ridge2, stack_test_ridge2),
    'Stack LGB+feats':   (stack_rmsle_lgb,    meta_test_lgb),
}
best_initial_name = min(strategies_initial, key=lambda k: strategies_initial[k][0])
best_initial_rmsle = strategies_initial[best_initial_name][0]
best_initial_log   = strategies_initial[best_initial_name][1]
print(f"\n  Best initial strategy: {best_initial_name} (CV RMSLE = {best_initial_rmsle:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: PSEUDO-LABELING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Pseudo-Labeling")
print("=" * 60)
print(f"  Rounds: {PSEUDO_ROUNDS} | Pseudo weight: {PSEUDO_WEIGHT}")
print(f"  Adding {len(test_fe)} pseudo-labeled test rows to {len(train_fe)} real rows (+{100*len(test_fe)/len(train_fe):.0f}%)")


def run_stacking_with_weights(train_combined, y_combined, test_fe_full,
                               sample_weights, lgb_p, cat_p, label=""):
    """
    Re-run full stacking with sample weights on the combined (train+pseudo) dataset.
    Returns (best_rmsle_cv, best_test_log_preds, individual_rmsles_dict).
    Only evaluated on the REAL training rows for an honest CV score.
    """
    n_real = len(train_fe)  # real rows are always first in the combined df

    # Factories with sample weights supported
    factories = {
        'lgb':   lambda: lgb.LGBMRegressor(**lgb_p),
        'cat':   lambda: CatBoostRegressor(**cat_p),
        'xgb':   lambda: xgb.XGBRegressor(**xgb_params),
        'rf':    lambda: RandomForestRegressor(
            n_estimators=300, max_depth=15, min_samples_leaf=3,
            max_features=0.6, random_state=RANDOM_STATE, n_jobs=-1),
        'ridge': lambda: Ridge(alpha=10.0),
    }

    # Apply target encoding on full combined set for test predictions
    prix_full_comb = np.expm1(y_combined.values)
    gm_full = prix_full_comb.mean()
    train_enc_comb = apply_target_encoding(train_combined, 'quartier',
                                            pd.Series(prix_full_comb, index=train_combined.index),
                                            gm_full)
    test_enc_comb  = apply_target_encoding(test_fe_full, 'quartier',
                                            pd.Series(prix_full_comb, index=train_combined.index),
                                            gm_full)

    X_full_comb = train_enc_comb[ALL_FEATURE_COLS].fillna(0)
    X_test_comb = test_enc_comb[ALL_FEATURE_COLS].fillna(0)
    sw = sample_weights  # numpy array, length = len(train_combined)

    oof_p, test_p = {}, {}
    # Only evaluate on real rows (first n_real)
    kf_pl = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + 7)

    for name, fn in factories.items():
        oof_arr = np.zeros(len(train_combined))
        test_folds = np.zeros((len(test_fe_full), N_FOLDS))
        for fold, (tr_idx, val_idx) in enumerate(kf_pl.split(X_full_comb)):
            X_tr = X_full_comb.iloc[tr_idx]
            X_val = X_full_comb.iloc[val_idx]
            y_tr  = y_combined.iloc[tr_idx]
            w_tr  = sw[tr_idx]

            m = fn()
            try:
                m.fit(X_tr, y_tr, sample_weight=w_tr)
            except TypeError:
                m.fit(X_tr, y_tr)  # Ridge doesn't use sample_weight in fit but RidgeCV does
            oof_arr[val_idx] = m.predict(X_val)
            test_folds[:, fold] = m.predict(X_test_comb)
        oof_p[name]  = oof_arr
        test_p[name] = test_folds.mean(axis=1)

    # CV score on real rows only
    ind_rmsles = {n: rmsle(y_combined.values[:n_real], oof_p[n][:n_real]) for n in factories}
    print(f"    {label} individual OOF (real rows only):")
    for n, s in sorted(ind_rmsles.items(), key=lambda x: x[1]):
        print(f"      {n:<10} {s:.4f}")

    # Stacking on real rows
    stack_df_pl   = pd.DataFrame(oof_p)
    stack_test_pl = pd.DataFrame(test_p)
    meta_oof_pl   = np.zeros(len(train_combined))
    meta_test_pl_folds = np.zeros((len(test_fe_full), N_FOLDS))
    kf_m = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + 99)
    for fold, (tr_idx, val_idx) in enumerate(kf_m.split(stack_df_pl)):
        m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        m.fit(stack_df_pl.iloc[tr_idx], y_combined.values[tr_idx],
              sample_weight=sw[tr_idx])
        meta_oof_pl[val_idx]              = m.predict(stack_df_pl.iloc[val_idx])
        meta_test_pl_folds[:, fold]       = m.predict(stack_test_pl)

    stack_test_out = meta_test_pl_folds.mean(axis=1)
    stack_cv = rmsle(y_combined.values[:n_real], meta_oof_pl[:n_real])

    # Also try weighted blend
    inv_r = {k: 1/v for k, v in ind_rmsles.items()}
    tot = sum(inv_r.values())
    w_bl = {k: v/tot for k, v in inv_r.items()}
    blend_test_out = sum(w_bl[k] * test_p[k] for k in factories)
    blend_cv = rmsle(y_combined.values[:n_real],
                     sum(w_bl[k] * oof_p[k][:n_real] for k in factories))

    best_cv = min(stack_cv, blend_cv)
    best_test = stack_test_out if stack_cv <= blend_cv else blend_test_out
    print(f"    {label} stacking CV (real rows): {stack_cv:.4f} | blend CV: {blend_cv:.4f}")
    return best_cv, best_test


# Initial pseudo-labels come from the best initial strategy
current_test_log = best_initial_log.copy()
best_pseudo_rmsle = best_initial_rmsle
best_pseudo_test  = best_initial_log.copy()

for rnd in range(1, PSEUDO_ROUNDS + 1):
    print(f"\n  ── Round {rnd} ──")

    # Build pseudo-labeled training set
    test_pseudo = test_fe.copy()
    test_pseudo['prix'] = np.expm1(current_test_log)
    combined_fe = pd.concat([train_fe, test_pseudo], ignore_index=True)
    y_pseudo_log = np.log1p(test_pseudo['prix'].values)
    y_combined   = pd.concat(
        [pd.Series(y.values), pd.Series(y_pseudo_log)], ignore_index=True
    )

    # Sample weights: 1.0 for real training rows, PSEUDO_WEIGHT for pseudo rows
    sw = np.concatenate([
        np.ones(len(train_fe)),
        np.full(len(test_fe), PSEUDO_WEIGHT)
    ])

    round_rmsle, round_test = run_stacking_with_weights(
        combined_fe, y_combined, test_fe,
        sw, lgb_params, cat_params,
        label=f"Round {rnd}"
    )

    print(f"  Round {rnd} CV RMSLE (real rows): {round_rmsle:.4f}"
          f"  (vs initial {best_initial_rmsle:.4f})")

    # Update pseudo-labels for next round; keep best result
    current_test_log = round_test
    if round_rmsle < best_pseudo_rmsle:
        best_pseudo_rmsle = round_rmsle
        best_pseudo_test  = round_test

print(f"\n  Best pseudo-labeling CV: {best_pseudo_rmsle:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: SELECT BEST OVERALL & OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — Selection & Submission")
print("=" * 60)

all_strategies = {
    'Initial Blending':          (blend_rmsle_val,    blend_test),
    'Initial Stack Ridge':       (stack_rmsle_ridge,  stack_test_ridge),
    'Initial Stack Ridge+feats': (stack_rmsle_ridge2, stack_test_ridge2),
    'Initial Stack LGB':         (stack_rmsle_lgb,    meta_test_lgb),
    'Pseudo-labeling':           (best_pseudo_rmsle,  best_pseudo_test),
}

best_name      = min(all_strategies, key=lambda k: all_strategies[k][0])
best_rmsle_val = all_strategies[best_name][0]
best_pred_log  = all_strategies[best_name][1]

print(f"\n  Best strategy: {best_name} (RMSLE CV = {best_rmsle_val:.4f})")

# Convert to original scale — floor at 100k MRU (minimum realistic price)
final_pred = np.maximum(np.expm1(best_pred_log), 100_000)
# V5: NO test-time per-quartier clipping — it was systematically wrong on small quartiers


submission = pd.DataFrame({
    'id':   test_fe['id'].values,
    'prix': final_pred.round(0).astype(int),
})
sub_path = os.path.join(OUT_DIR, 'submission_v7.csv')
submission.to_csv(sub_path, index=False)
print(f"\n  Submission: {sub_path} ({len(submission)} rows)")
print(f"  Prix min:    {submission['prix'].min():,} MRU")
print(f"  Prix max:    {submission['prix'].max():,} MRU")
print(f"  Prix median: {submission['prix'].median():,.0f} MRU")

# Save also as submission_improved.csv for API compatibility
sub_path2 = os.path.join(OUT_DIR, 'submission_improved.csv')
submission.to_csv(sub_path2, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON TABLE")
print("=" * 60)
print(f"\n  {'Model':<35} {'RMSLE CV':>10}")
print(f"  {'-'*47}")
print(f"  {'V1 RF (34 features)':<35} {'0.6576':>10}")
print(f"  {'V2 Stacking (77 features)':<35} {'0.6263':>10}")
print(f"  {'V4 Fixed stacking (honest CV)':<35} {'~0.608':>10}")
print(f"  {'-'*47}")
for name, score in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {name:<35} {score:>10.4f}")
for sname, (srmsle, _) in sorted(all_strategies.items(), key=lambda x: x[1][0]):
    marker = " ← BEST" if sname == best_name else ""
    print(f"  {sname:<35} {srmsle:>10.4f}{marker}")
print(f"\n  V4 → V5 improvement: {0.608 - best_rmsle_val:+.4f} RMSLE")
print(f"  Gap to target (0.52621): {best_rmsle_val - 0.52621:+.4f}")

# Save best individual model
best_single = min(model_rmsles.items(), key=lambda x: x[1])
print(f"\n  Best individual model: {best_single[0]} (RMSLE={best_single[1]:.4f})")
best_model_instance = base_model_factories[best_single[0]]()
best_model_instance.fit(X_train_full, y)
joblib.dump(best_model_instance, os.path.join(MODEL_DIR, 'housing_model_improved.pkl'))
joblib.dump(ALL_FEATURE_COLS, os.path.join(MODEL_DIR, 'features_improved.pkl'))
print("  Model saved: model/housing_model_improved.pkl")

# Update geo_meta.json
with open(GEO_META, encoding='utf-8') as f:
    _geo_out = json.load(f)
_geo_out['cv_rmsle'] = round(best_rmsle_val, 6)
with open(GEO_META, 'w', encoding='utf-8') as f:
    json.dump(_geo_out, f, ensure_ascii=False, indent=2)
print(f"  geo_meta.json updated with cv_rmsle = {best_rmsle_val:.4f}")
print("\nDone!")