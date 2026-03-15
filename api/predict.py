"""
predict.py — Logique de prédiction isolée
Reconstruit le vecteur de 34 features à partir de l'input utilisateur.
"""
import re
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from datetime import datetime

from config import (
    QUARTIER_GPS, CENTRE_GPS, AEROPORT_GPS, PLAGE_GPS,
    QUARTIER_POI, QUARTIER_TAILLE_RUE, DEFAULT_TAILLE_RUE,
    DATE_REF, MRU_TO_EUR,
)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance en km entre deux points GPS."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def build_feature_vector(data: dict, feature_cols: list,
                          target_enc: dict, freq_enc: dict,
                          feature_medians: dict) -> np.ndarray:
    """
    Construit le vecteur de features dans le bon ordre à partir du JSON d'entrée.

    Args:
        data: dict reçu dans la requête POST /api/predict
        feature_cols: liste ordonnée des 34 features
        target_enc: dict quartier → prix moyen (calculé sur le train)
        freq_enc: dict quartier → fréquence relative
        feature_medians: dict feature → médiane du train (fallback)

    Returns:
        np.ndarray de shape (1, nb_features)
    """
    quartier = data.get('quartier', 'Teyarett')
    surface  = float(data.get('surface_m2', 200))
    nb_ch    = float(data.get('nb_chambres', 3))
    nb_sal   = float(data.get('nb_salons', 1))
    nb_sdb_val = data.get('nb_sdb', None)
    desc     = str(data.get('description', '')).lower()

    # GPS
    lat, lon = QUARTIER_GPS.get(quartier, QUARTIER_GPS['Teyarett'])

    # Distances
    dist_centre   = haversine(lat, lon, *CENTRE_GPS)
    dist_aeroport = haversine(lat, lon, *AEROPORT_GPS)
    dist_plage    = haversine(lat, lon, *PLAGE_GPS)

    # POI
    poi = QUARTIER_POI.get(quartier, {'ecoles': 4, 'mosquees': 6, 'commerces': 8, 'hopitaux': 1})
    nb_ecoles    = poi['ecoles']
    nb_mosquees  = poi['mosquees']
    nb_commerces = poi['commerces']
    nb_hopitaux  = poi['hopitaux']
    nb_total_poi = nb_ecoles + nb_mosquees + nb_commerces + nb_hopitaux

    # has_sdb_info : 1 si renseigné par l'utilisateur
    if nb_sdb_val is None or nb_sdb_val == '':
        has_sdb_info = 0
        nb_sdb = feature_medians.get('nb_sdb', 1.0)
    else:
        has_sdb_info = 1
        nb_sdb = float(nb_sdb_val)

    # Features texte depuis description
    combined = desc + ' ' + str(data.get('caracteristiques', '')).lower()
    has_garage       = int(data.get('has_garage', 0))
    has_titre_foncier = int(data.get('has_titre_foncier', 0))
    has_camera       = int('cam' in combined)
    nb_balcons       = _extract_balcons(combined)
    taille_rue       = QUARTIER_TAILLE_RUE.get(quartier, DEFAULT_TAILLE_RUE)

    has_piscine = int(data.get('has_piscine', 0) or 'piscine' in combined)
    has_clim    = int(data.get('has_clim', 0) or 'clim' in combined or 'climatisation' in combined)
    has_meuble  = int('meubl' in combined)
    is_luxe     = int('luxe' in combined or 'standing' in combined)
    is_renove   = int('renov' in combined or 'neuf' in combined or 'nouveau' in combined)
    has_arabic  = int(bool(re.search(r'[\u0600-\u06FF]', data.get('description', ''))))

    desc_len        = len(data.get('description', ''))
    desc_word_count = len(data.get('description', '').split())

    # Variables dérivées
    nb_pieces_total   = nb_ch + nb_sal
    surface_par_piece = surface / max(nb_pieces_total, 1)
    log_surface       = np.log1p(surface)

    # Âge de l'annonce (on utilise aujourd'hui comme date de publication fictive = 0 jours)
    age_annonce_jours = feature_medians.get('age_annonce_jours', 365.0)

    # Encodage quartier
    global_mean = float(np.mean(list(target_enc.values()))) if target_enc else 2600000.0
    quartier_target_enc = target_enc.get(quartier, global_mean)
    quartier_freq       = freq_enc.get(quartier, 1 / 8)

    # Construire le dict complet
    feat_dict = {
        'surface_m2':          surface,
        'nb_chambres':         nb_ch,
        'nb_salons':           nb_sal,
        'nb_sdb':              nb_sdb,
        'has_sdb_info':        has_sdb_info,
        'latitude':            lat,
        'longitude':           lon,
        'dist_centre_km':      dist_centre,
        'dist_aeroport_km':    dist_aeroport,
        'dist_plage_km':       dist_plage,
        'nb_ecoles_1km':       nb_ecoles,
        'nb_mosquees_1km':     nb_mosquees,
        'nb_commerces_1km':    nb_commerces,
        'nb_hopitaux_1km':     nb_hopitaux,
        'nb_total_pois_1km':   nb_total_poi,
        'nb_pieces_total':     nb_pieces_total,
        'has_garage':          has_garage,
        'has_titre_foncier':   has_titre_foncier,
        'has_camera':          has_camera,
        'nb_balcons':          nb_balcons,
        'taille_rue':          taille_rue,
        'desc_len':            desc_len,
        'desc_word_count':     desc_word_count,
        'has_piscine':         has_piscine,
        'has_clim':            has_clim,
        'has_meuble':          has_meuble,
        'is_luxe':             is_luxe,
        'is_renove':           is_renove,
        'has_arabic':          has_arabic,
        'age_annonce_jours':   age_annonce_jours,
        'surface_par_piece':   surface_par_piece,
        'log_surface':         log_surface,
        'quartier_target_enc': quartier_target_enc,
        'quartier_freq':       quartier_freq,
    }

    # Remplir les features manquantes par la médiane du train
    for col in feature_cols:
        if col not in feat_dict or feat_dict[col] is None:
            feat_dict[col] = feature_medians.get(col, 0.0)

    # Retourner dans le bon ordre
    return np.array([[feat_dict[col] for col in feature_cols]])


def predict_price(model, X: np.ndarray) -> dict:
    """
    Effectue la prédiction et calcule l'intervalle de confiance.

    Returns:
        dict avec prix_estime, intervalle (bas/haut)
    """
    log_prix = model.predict(X)[0]
    prix = float(np.expm1(log_prix))
    prix = max(prix, 100_000)

    # Intervalle via les arbres individuels (Random Forest)
    if hasattr(model, 'estimators_'):
        tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
        tree_prix  = np.expm1(tree_preds)
        intervalle = {
            'bas':  float(max(np.percentile(tree_prix, 10), 100_000)),
            'haut': float(np.percentile(tree_prix, 90)),
        }
    else:
        intervalle = {
            'bas':  prix * 0.80,
            'haut': prix * 1.20,
        }

    return {'prix': prix, 'intervalle': intervalle}


# ── Helpers privés ─────────────────────────────────────────────────────────────

def _extract_balcons(text: str) -> int:
    m = re.search(r'(\d+)\s*balcon', text)
    return int(m.group(1)) if m else 0
