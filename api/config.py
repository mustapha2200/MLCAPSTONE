"""
config.py — Configuration centralisée de l'API
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')

MODEL_DIR = os.path.join(ROOT_DIR, 'model')
DATA_DIR  = os.path.join(ROOT_DIR, 'data')

MODEL_PATH    = os.path.join(MODEL_DIR, 'housing_model.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'features.pkl')
TRAIN_CSV     = os.path.join(DATA_DIR, 'raw', 'kaggle_train.csv')

PORT = int(os.environ.get('PORT', 5000))
DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

# Coordonnées GPS des quartiers (hardcodées pour l'inférence offline)
QUARTIER_GPS = {
    'Tevragh Zeina': (18.1036, -15.9785),
    'Ksar':          (18.0866, -15.9750),
    'Arafat':        (18.0550, -15.9610),
    'Dar Naim':      (18.1200, -15.9450),
    'Toujounine':    (18.0680, -15.9350),
    'Sebkha':        (18.0730, -15.9870),
    'Riyadh':        (18.0850, -15.9550),
    'Teyarett':      (18.0950, -15.9700),
}

QUARTIER_CARACTERE = {
    'Tevragh Zeina': 'Quartier huppé, ambassades, villas',
    'Ksar':          'Centre historique',
    'Teyarett':      'Centre, mixte',
    'Sebkha':        'Commercial, marchés',
    'Arafat':        'Populaire, dense',
    'Dar Naim':      'Résidentiel, expansion',
    'Toujounine':    'Périphérie, récent',
    'Riyadh':        'Résidentiel moyen',
}

# Points de référence
CENTRE_GPS    = (18.0866, -15.9750)
AEROPORT_GPS  = (18.0987, -15.9476)
PLAGE_GPS     = (18.0580, -16.0200)

# POI approximatifs par quartier
QUARTIER_POI = {
    'Tevragh Zeina': {'ecoles': 8,  'mosquees': 6,  'commerces': 15, 'hopitaux': 4},
    'Ksar':          {'ecoles': 6,  'mosquees': 10, 'commerces': 20, 'hopitaux': 3},
    'Teyarett':      {'ecoles': 5,  'mosquees': 8,  'commerces': 12, 'hopitaux': 2},
    'Sebkha':        {'ecoles': 4,  'mosquees': 7,  'commerces': 25, 'hopitaux': 2},
    'Arafat':        {'ecoles': 6,  'mosquees': 12, 'commerces': 10, 'hopitaux': 1},
    'Dar Naim':      {'ecoles': 4,  'mosquees': 5,  'commerces': 8,  'hopitaux': 1},
    'Riyadh':        {'ecoles': 3,  'mosquees': 4,  'commerces': 6,  'hopitaux': 1},
    'Toujounine':    {'ecoles': 3,  'mosquees': 6,  'commerces': 5,  'hopitaux': 0},
}

# Taille de rue médiane par quartier (proxy)
QUARTIER_TAILLE_RUE = {
    'Tevragh Zeina': 12.0,
    'Ksar':          8.0,
    'Teyarett':      8.0,
    'Sebkha':        8.0,
    'Arafat':        6.0,
    'Dar Naim':      8.0,
    'Toujounine':    6.0,
    'Riyadh':        8.0,
}

DEFAULT_TAILLE_RUE = 8.0

MODEL_INFO = {
    'nom': 'Random Forest',
    'rmsle': 0.6576,
    'r2': 0.564,
    'nb_features': 34,
}

MRU_TO_EUR = 1 / 400
DATE_REF = '2026-03-02'
