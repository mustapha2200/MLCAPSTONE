"""
app.py — API Flask principale
Prédiction des prix immobiliers à Nouakchott
Aucune donnée hardcodée — tout est chargé depuis les fichiers de données et le modèle.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    MODEL_PATH, FEATURES_PATH, TRAIN_CSV, GEO_META_PATH,
    PORT, DEBUG, QUARTIER_GPS, QUARTIER_CARACTERE, QUARTIER_POI,
    CENTRE_GPS, AEROPORT_GPS, PLAGE_GPS, MRU_TO_EUR,
)
from predict import build_feature_vector, predict_price

app = Flask(__name__)
CORS(app)

# ── Chargement au démarrage ────────────────────────────────────────────────────

print("Chargement du modèle...")
model        = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURES_PATH)
print(f"Modèle chargé : {len(feature_cols)} features")

print("Chargement des données train...")
train = pd.read_csv(TRAIN_CSV)
train['quartier'] = train['quartier'].str.strip()

# Encodages calculés sur le train (pas hardcodés)
target_enc = train.groupby('quartier')['prix'].mean().to_dict()
freq_enc   = train['quartier'].value_counts(normalize=True).to_dict()

# Médianes des features pour le fallback
feature_medians = {}
enriched_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'enriched_data.csv')
if os.path.exists(enriched_path):
    enriched = pd.read_csv(enriched_path)
    for col in feature_cols:
        if col in enriched.columns:
            med = enriched[col].median()
            feature_medians[col] = float(med) if not pd.isna(med) else 0.0
        else:
            feature_medians[col] = 0.0
else:
    feature_medians = {col: 0.0 for col in feature_cols}

# Stats par quartier — calculées depuis le train
quartier_stats = (
    train.groupby('quartier')
    .agg(
        nb_annonces=('prix', 'count'),
        prix_median=('prix', 'median'),
        prix_moyen=('prix', 'mean'),
        surface_mediane=('surface_m2', 'median'),
    )
    .reset_index()
    .to_dict(orient='records')
)

# Infos modèle — dérivées du modèle chargé, pas hardcodées
model_type = type(model).__name__
nb_features = len(feature_cols)

# Feature importances — extraites du modèle
feature_importances = {}
if hasattr(model, 'feature_importances_'):
    total = model.feature_importances_.sum()
    for name, imp in zip(feature_cols, model.feature_importances_):
        feature_importances[name] = round(float(imp / total), 6)
    feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

# RMSLE CV — lu depuis geo_meta.json s'il existe, sinon None
with open(GEO_META_PATH, encoding='utf-8') as f:
    _geo_meta = json.load(f)
cv_rmsle         = _geo_meta.get('cv_rmsle')
cv_r2            = _geo_meta.get('cv_r2')
model_comparison = _geo_meta.get('model_comparison', [])

print("API prête.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def root():
    return jsonify({'status': 'ok', 'service': 'ImmobilierNK API'})


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status':   'ok',
        'model':    model_type,
        'features': nb_features,
        'rmsle':    cv_rmsle,
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'Corps JSON vide'}), 400

    quartier = data.get('quartier', 'Teyarett')
    surface  = float(data.get('surface_m2', 200))

    try:
        X      = build_feature_vector(data, feature_cols, target_enc, freq_enc, feature_medians)
        result = predict_price(model, X)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    prix    = result['prix']
    prix_m2 = prix / max(surface, 1)

    qs = next((s for s in quartier_stats if s['quartier'] == quartier), None)
    if qs:
        prix_m2_q  = qs['prix_moyen'] / max(float(qs['surface_mediane']), 1)
        comparable = {
            'prix_median_quartier':      int(qs['prix_median']),
            'nb_annonces_quartier':      int(qs['nb_annonces']),
            'surface_mediane_quartier':  int(qs['surface_mediane']),
        }
    else:
        prix_m2_q  = 0
        comparable = {}

    return jsonify({
        'prix_estime':            int(round(prix)),
        'prix_estime_eur':        int(round(prix * MRU_TO_EUR)),
        'prix_m2':                int(round(prix_m2)),
        'prix_m2_quartier_moyen': int(round(prix_m2_q)),
        'intervalle': {
            'bas':  int(round(result['intervalle']['bas'])),
            'haut': int(round(result['intervalle']['haut'])),
        },
        'quartier':   quartier,
        'comparable': comparable,
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    prix_vals = train['prix'].values / 1e6
    bins  = [0, 0.5, 1, 1.5, 2, 3, 5, 8, 12, 20, 55]
    counts, _ = np.histogram(prix_vals, bins=bins)

    quartiers_list = []
    for row in quartier_stats:
        q = row['quartier']
        lat, lon = QUARTIER_GPS.get(q, (train['surface_m2'].mean(), 0))
        quartiers_list.append({
            'nom':             q,
            'prix_median':     int(row['prix_median']),
            'prix_moyen':      int(row['prix_moyen']),
            'nb_annonces':     int(row['nb_annonces']),
            'surface_mediane': int(row['surface_mediane']),
            'latitude':        lat,
            'longitude':       lon,
            'caractere':       QUARTIER_CARACTERE.get(q, ''),
        })

    return jsonify({
        'total_annonces':  int(len(train)),
        'prix_median':     int(train['prix'].median()),
        'prix_moyen':      int(train['prix'].mean()),
        'surface_mediane': int(train['surface_m2'].median()),
        'quartiers':       quartiers_list,
        'distribution_prix': {
            'bins':   [round(b, 1) for b in bins],
            'counts': counts.tolist(),
        },
        'model_info': {
            'nom':        model_type,
            'rmsle':      cv_rmsle,
            'r2':         cv_r2,
            'nb_features': nb_features,
        },
    })


@app.route('/api/quartiers', methods=['GET'])
def quartiers():
    result = []
    for row in quartier_stats:
        q = row['quartier']
        lat, lon = QUARTIER_GPS.get(q, (18.0866, -15.975))
        result.append({
            'nom':             q,
            'latitude':        lat,
            'longitude':       lon,
            'prix_median':     int(row['prix_median']),
            'nb_annonces':     int(row['nb_annonces']),
            'surface_mediane': int(row['surface_mediane']),
            'caractere':       QUARTIER_CARACTERE.get(q, ''),
        })
    result.sort(key=lambda x: x['prix_median'], reverse=True)
    return jsonify(result)


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Retourne les infos du modèle et les feature importances calculées depuis le pkl."""
    top_n = int(request.args.get('top', 20))
    top_features = list(feature_importances.items())[:top_n]
    return jsonify({
        'model':            model_type,
        'nb_features':      nb_features,
        'rmsle':            cv_rmsle,
        'r2':               cv_r2,
        'feature_importances': [
            {'name': name, 'importance': imp} for name, imp in top_features
        ],
        'model_comparison': model_comparison,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
