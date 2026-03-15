"""
config.py — Configuration centralisée de l'API
Toutes les données géographiques sont chargées depuis data/processed/geo_meta.json
"""
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')

MODEL_DIR = os.path.join(ROOT_DIR, 'model')
DATA_DIR  = os.path.join(ROOT_DIR, 'data')

MODEL_PATH    = os.path.join(MODEL_DIR, 'housing_model.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'features.pkl')
TRAIN_CSV     = os.path.join(DATA_DIR, 'raw', 'kaggle_train.csv')
GEO_META_PATH = os.path.join(DATA_DIR, 'processed', 'geo_meta.json')

PORT  = int(os.environ.get('PORT', 5000))
DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

# Charger les métadonnées géographiques depuis le fichier JSON
with open(GEO_META_PATH, encoding='utf-8') as f:
    _geo_meta = json.load(f)

QUARTIER_META = _geo_meta['quartiers']
REFERENCE_POINTS = _geo_meta['reference_points']

# Dériver les tables utilisées dans l'API
QUARTIER_GPS = {q: (v['lat'], v['lon']) for q, v in QUARTIER_META.items()}
QUARTIER_CARACTERE = {q: v['caractere'] for q, v in QUARTIER_META.items()}
QUARTIER_POI = {
    q: {
        'ecoles':    v['poi']['ecoles'],
        'mosquees':  v['poi']['mosquees'],
        'commerces': v['poi']['commerces'],
        'hopitaux':  v['poi']['hopitaux'],
    }
    for q, v in QUARTIER_META.items()
}

CENTRE_GPS   = (REFERENCE_POINTS['centre']['lat'],   REFERENCE_POINTS['centre']['lon'])
AEROPORT_GPS = (REFERENCE_POINTS['aeroport']['lat'], REFERENCE_POINTS['aeroport']['lon'])
PLAGE_GPS    = (REFERENCE_POINTS['plage']['lat'],    REFERENCE_POINTS['plage']['lon'])

QUARTIER_TAILLE_RUE = _geo_meta.get('taille_rue', {})
DEFAULT_TAILLE_RUE  = _geo_meta.get('default_taille_rue', 6.5)

MRU_TO_EUR = 1 / 400

from datetime import date
DATE_REF = date(2026, 3, 2)

# Infos modèle — lues depuis les fichiers pkl au démarrage de app.py
# (MODEL_INFO est construit dynamiquement dans app.py)
