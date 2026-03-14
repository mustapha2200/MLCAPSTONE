"""
geo_enrichment.py — Géocodage Nominatim + POI Overpass API + distances
Projet Capstone : Prédiction des Prix Immobiliers à Nouakchott

Flux :
  1. geocode_quartiers()  → Nominatim (OSM) : nom quartier → (lat, lon)
  2. fetch_poi_counts()   → Overpass API   : POI dans un rayon de 1 km
  3. Résultats mis en cache dans data/processed/geo_cache.json
  4. Si API inaccessible, fallback sur les coordonnées hardcodées
"""

import json
import time
import os
import logging
from math import radians, cos, sin, asin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

logger = logging.getLogger(__name__)

# ── Cache ─────────────────────────────────────────────────────────────────────
_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "geo_cache.json"

# ── Fallback hardcodé (utilisé si Nominatim échoue) ──────────────────────────
_GPS_FALLBACK = {
    'Tevragh Zeina': (18.1036, -15.9785),
    'Sebkha':        (18.0730, -15.9870),
    'Ksar':          (18.0866, -15.9750),
    'Teyarett':      (18.0950, -15.9700),
    'Dar Naim':      (18.1200, -15.9450),
    'Arafat':        (18.0550, -15.9610),
    'Toujounine':    (18.0680, -15.9350),
    'Riyadh':        (18.0850, -15.9550),
    'El Mina':       (18.0580, -16.0200),
}

# Noms de recherche Nominatim (plus précis que les noms courts)
_NOMINATIM_QUERIES = {
    'Tevragh Zeina': 'Tevragh Zeina, Nouakchott, Mauritanie',
    'Sebkha':        'Sebkha, Nouakchott, Mauritanie',
    'Ksar':          'Ksar, Nouakchott, Mauritanie',
    'Teyarett':      'Teyarett, Nouakchott, Mauritanie',
    'Dar Naim':      'Dar Naim, Nouakchott, Mauritanie',
    'Arafat':        'Arafat, Nouakchott, Mauritanie',
    'Toujounine':    'Toujounine, Nouakchott, Mauritanie',
    'Riyadh':        'Riyadh, Nouakchott, Mauritanie',
    'El Mina':       'El Mina, Nouakchott, Mauritanie',
}

# ── Points de référence ───────────────────────────────────────────────────────
CENTRE_VILLE = (18.0866, -15.9750)   # Ksar
AEROPORT     = (18.0987, -15.9476)   # Aéroport Oumtounsy
PLAGE        = (18.0580, -16.0200)   # El Mina


# ── Utilitaires ───────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance haversine en km entre deux points GPS."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def _load_cache() -> dict:
    """Charge le cache JSON depuis le disque."""
    if _CACHE_PATH.exists():
        with open(_CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    """Sauvegarde le cache JSON sur le disque."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ── 1. Géocodage Nominatim ────────────────────────────────────────────────────

def geocode_quartiers(quartiers: list,
                      user_agent: str = "mauritania_housing_capstone",
                      delay: float = 1.1) -> dict:
    """
    Géocode une liste de noms de quartiers via Nominatim (OpenStreetMap).

    Paramètres
    ----------
    quartiers  : liste des noms de quartiers à géocoder
    user_agent : identifiant requis par la politique d'utilisation Nominatim
    delay      : pause entre chaque requête (min 1 s imposé par OSM)

    Retourne
    --------
    dict : {quartier: (latitude, longitude)}
           Valeurs de fallback utilisées si Nominatim ne trouve pas le quartier.
    """
    cache = _load_cache()
    gps_cache = cache.get("gps", {})

    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    result = {}
    new_entries = False

    for quartier in quartiers:
        # Déjà en cache ?
        if quartier in gps_cache:
            coords = gps_cache[quartier]
            result[quartier] = tuple(coords)
            logger.info(f"[Nominatim] Cache hit : {quartier} → {coords}")
            continue

        query = _NOMINATIM_QUERIES.get(quartier, f"{quartier}, Nouakchott, Mauritanie")
        try:
            time.sleep(delay)   # Respect du fair-use Nominatim (1 req/s)
            location = geolocator.geocode(query)

            if location:
                coords = (round(location.latitude, 6), round(location.longitude, 6))
                logger.info(f"[Nominatim] {quartier} → {coords}  ({location.address})")
            else:
                # Deuxième tentative avec requête simplifiée
                time.sleep(delay)
                location2 = geolocator.geocode(f"{quartier}, Mauritanie")
                if location2:
                    coords = (round(location2.latitude, 6), round(location2.longitude, 6))
                    logger.info(f"[Nominatim] {quartier} (2e tentative) → {coords}")
                else:
                    coords = _GPS_FALLBACK.get(quartier, (np.nan, np.nan))
                    logger.warning(f"[Nominatim] {quartier} introuvable → fallback {coords}")

        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            coords = _GPS_FALLBACK.get(quartier, (np.nan, np.nan))
            logger.warning(f"[Nominatim] Erreur pour {quartier} : {e} → fallback")

        gps_cache[quartier] = list(coords)
        result[quartier] = coords
        new_entries = True

    if new_entries:
        cache["gps"] = gps_cache
        _save_cache(cache)
        print(f"[Nominatim] Cache GPS mis à jour → {_CACHE_PATH}")

    return result


# ── 2. Overpass API — comptage des POI ────────────────────────────────────────

# Définition des types de POI et leurs requêtes Overpass
_POI_QUERIES = {
    "ecoles": """
        node["amenity"~"school|kindergarten|college|university"](around:{radius},{lat},{lon});
        way["amenity"~"school|kindergarten|college|university"](around:{radius},{lat},{lon});
    """,
    "mosquees": """
        node["amenity"="place_of_worship"]["religion"="muslim"](around:{radius},{lat},{lon});
        way["amenity"="place_of_worship"]["religion"="muslim"](around:{radius},{lat},{lon});
    """,
    "commerces": """
        node["shop"](around:{radius},{lat},{lon});
        way["shop"](around:{radius},{lat},{lon});
        node["amenity"~"marketplace|supermarket"](around:{radius},{lat},{lon});
    """,
    "hopitaux": """
        node["amenity"~"hospital|clinic|doctors|pharmacy|health_centre"](around:{radius},{lat},{lon});
        way["amenity"~"hospital|clinic|doctors|pharmacy|health_centre"](around:{radius},{lat},{lon});
    """,
}

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _build_overpass_query(lat: float, lon: float,
                           poi_type: str, radius: int = 1000) -> str:
    """Construit une requête Overpass QL pour compter les POI."""
    inner = _POI_QUERIES[poi_type].format(lat=lat, lon=lon, radius=radius)
    return f"""
[out:json][timeout:30];
(
{inner}
);
out count;
"""


def _query_overpass(query: str, retries: int = 3, delay: float = 2.0) -> int:
    """
    Exécute une requête Overpass et retourne le nombre d'éléments trouvés.
    Retry automatique en cas d'erreur réseau (429 ou timeout).
    """
    for attempt in range(retries):
        try:
            resp = requests.post(
                _OVERPASS_URL,
                data={"data": query},
                timeout=35,
                headers={"User-Agent": "mauritania_housing_capstone"}
            )
            if resp.status_code == 429:
                wait = delay * (attempt + 1) * 10
                logger.warning(f"[Overpass] Rate limit → attente {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Overpass "out count" → élément avec tag "total"
            elements = data.get("elements", [])
            if elements and "tags" in elements[0]:
                return int(elements[0]["tags"].get("total", 0))
            return len(elements)

        except requests.exceptions.RequestException as e:
            logger.warning(f"[Overpass] Tentative {attempt+1}/{retries} échouée : {e}")
            time.sleep(delay * (attempt + 1))

    logger.error("[Overpass] Toutes les tentatives ont échoué, retour 0")
    return 0


def fetch_poi_counts(gps_dict: dict,
                     radius: int = 1000,
                     delay: float = 2.0) -> dict:
    """
    Récupère les comptages POI pour chaque quartier via l'API Overpass.

    Paramètres
    ----------
    gps_dict : {quartier: (lat, lon)} — issu de geocode_quartiers()
    radius   : rayon de recherche en mètres (défaut 1 000 m = 1 km)
    delay    : pause entre chaque requête Overpass (pour respecter le rate limit)

    Retourne
    --------
    dict : {quartier: {'ecoles': int, 'mosquees': int, 'commerces': int, 'hopitaux': int}}
    """
    cache = _load_cache()
    poi_cache = cache.get("poi", {})

    result = {}
    new_entries = False

    for quartier, (lat, lon) in gps_dict.items():
        cache_key = f"{quartier}_{radius}m"

        if cache_key in poi_cache:
            result[quartier] = poi_cache[cache_key]
            logger.info(f"[Overpass] Cache hit : {quartier} → {poi_cache[cache_key]}")
            continue

        if np.isnan(lat) or np.isnan(lon):
            logger.warning(f"[Overpass] GPS manquant pour {quartier} → POI = 0")
            result[quartier] = {k: 0 for k in _POI_QUERIES}
            continue

        print(f"  [Overpass] Requête POI pour {quartier} ({lat:.4f}, {lon:.4f})…")
        counts = {}

        for poi_type in _POI_QUERIES:
            query = _build_overpass_query(lat, lon, poi_type, radius)
            count = _query_overpass(query)
            counts[poi_type] = count
            time.sleep(delay)   # Respect du fair-use Overpass

        print(f"  → {counts}")
        result[quartier] = counts
        poi_cache[cache_key] = counts
        new_entries = True

    if new_entries:
        cache["poi"] = poi_cache
        _save_cache(cache)
        print(f"[Overpass] Cache POI mis à jour → {_CACHE_PATH}")

    return result


# ── 3. Application au DataFrame ───────────────────────────────────────────────

def build_geo_tables(quartiers: list,
                     radius: int = 1000,
                     nominatim_delay: float = 1.1,
                     overpass_delay: float = 2.0) -> tuple[dict, dict]:
    """
    Orchestre le géocodage Nominatim et la récupération POI Overpass.

    Retourne
    --------
    gps_dict : {quartier: (lat, lon)}
    poi_dict : {quartier: {'ecoles': int, 'mosquees': int, 'commerces': int, 'hopitaux': int}}
    """
    print("\n[GEO] Étape 1 — Géocodage Nominatim…")
    gps_dict = geocode_quartiers(quartiers, delay=nominatim_delay)

    print("\n[GEO] Étape 2 — Comptage POI via Overpass API…")
    poi_dict = fetch_poi_counts(gps_dict, radius=radius, delay=overpass_delay)

    return gps_dict, poi_dict


def add_geo_features(df: pd.DataFrame, gps_dict: dict) -> pd.DataFrame:
    """Ajoute latitude, longitude et distances haversine vers les 3 repères."""
    df = df.copy()

    df['latitude']  = df['quartier'].map(
        lambda q: gps_dict.get(q, _GPS_FALLBACK.get(q, (np.nan, np.nan)))[0])
    df['longitude'] = df['quartier'].map(
        lambda q: gps_dict.get(q, _GPS_FALLBACK.get(q, (np.nan, np.nan)))[1])

    def _dist(row, ref):
        if pd.isna(row['latitude']):
            return np.nan
        return haversine(row['latitude'], row['longitude'], ref[0], ref[1])

    df['dist_centre_km']   = df.apply(_dist, ref=CENTRE_VILLE, axis=1)
    df['dist_aeroport_km'] = df.apply(_dist, ref=AEROPORT,     axis=1)
    df['dist_plage_km']    = df.apply(_dist, ref=PLAGE,        axis=1)

    return df


# POI hardcodé de secours (utilisé si Overpass retourne 0 pour tous les types)
_POI_FALLBACK = {
    'Tevragh Zeina': {'ecoles': 8,  'mosquees': 6,  'commerces': 15, 'hopitaux': 4},
    'Ksar':          {'ecoles': 6,  'mosquees': 10, 'commerces': 20, 'hopitaux': 3},
    'Teyarett':      {'ecoles': 5,  'mosquees': 8,  'commerces': 12, 'hopitaux': 2},
    'Sebkha':        {'ecoles': 4,  'mosquees': 7,  'commerces': 25, 'hopitaux': 2},
    'Arafat':        {'ecoles': 6,  'mosquees': 12, 'commerces': 10, 'hopitaux': 1},
    'Dar Naim':      {'ecoles': 4,  'mosquees': 5,  'commerces': 8,  'hopitaux': 1},
    'Riyadh':        {'ecoles': 3,  'mosquees': 4,  'commerces': 6,  'hopitaux': 1},
    'Toujounine':    {'ecoles': 3,  'mosquees': 6,  'commerces': 5,  'hopitaux': 0},
    'El Mina':       {'ecoles': 4,  'mosquees': 8,  'commerces': 8,  'hopitaux': 1},
}


def _resolve_poi(quartier: str, poi_dict: dict, default_poi: dict) -> dict:
    """
    Retourne les POI d'un quartier.
    Si Overpass a retourné 0 pour tous les types (geocodage imprécis probable),
    on utilise les valeurs de référence hardcodées.
    """
    counts = poi_dict.get(quartier, default_poi)
    total = sum(counts.values())
    if total == 0 and quartier in _POI_FALLBACK:
        # Overpass a retourné 0 (coordonnées OSM imprécises pour ce quartier)
        return _POI_FALLBACK[quartier]
    return counts


def add_poi_features(df: pd.DataFrame, poi_dict: dict) -> pd.DataFrame:
    """Ajoute les comptages POI réels depuis Overpass (avec fallback si 0)."""
    df = df.copy()

    # Valeur par défaut = médiane des quartiers avec des POI non nuls
    def _median_poi(key: str) -> int:
        vals = [poi_dict[q][key] for q in poi_dict
                if key in poi_dict[q] and sum(poi_dict[q].values()) > 0]
        return int(np.median(vals)) if vals else 0

    default_poi = {k: _median_poi(k) for k in ['ecoles', 'mosquees', 'commerces', 'hopitaux']}

    for feature, key in [
        ('nb_ecoles_1km',    'ecoles'),
        ('nb_mosquees_1km',  'mosquees'),
        ('nb_commerces_1km', 'commerces'),
        ('nb_hopitaux_1km',  'hopitaux'),
    ]:
        df[feature] = df['quartier'].map(
            lambda q, k=key: _resolve_poi(q, poi_dict, default_poi)[k]
        )

    df['nb_total_pois_1km'] = (
        df['nb_ecoles_1km'] + df['nb_mosquees_1km'] +
        df['nb_commerces_1km'] + df['nb_hopitaux_1km']
    )

    return df


# ── Point d'entrée principal ──────────────────────────────────────────────────

# Tables en mémoire (peuplées lors du premier appel à enrich_geo)
_GPS_TABLE: dict = {}
_POI_TABLE: dict = {}


def enrich_geo(df: pd.DataFrame,
               radius: int = 1000,
               force_refresh: bool = False) -> pd.DataFrame:
    """
    Pipeline complet :
      1. Géocode les quartiers via Nominatim (avec cache)
      2. Récupère les POI via Overpass (avec cache)
      3. Ajoute toutes les features géographiques au DataFrame

    Paramètres
    ----------
    df            : DataFrame avec colonne 'quartier'
    radius        : rayon POI en mètres
    force_refresh : si True, ignore le cache et refait toutes les requêtes
    """
    global _GPS_TABLE, _POI_TABLE

    if force_refresh:
        # Vider le cache disque
        if _CACHE_PATH.exists():
            _CACHE_PATH.unlink()
        _GPS_TABLE = {}
        _POI_TABLE = {}

    quartiers = df['quartier'].dropna().unique().tolist()

    # Construire les tables GPS et POI si pas encore en mémoire
    if not _GPS_TABLE or not _POI_TABLE:
        _GPS_TABLE, _POI_TABLE = build_geo_tables(quartiers, radius=radius)

    df = add_geo_features(df, _GPS_TABLE)
    df = add_poi_features(df, _POI_TABLE)

    return df


# ── CLI — affiche les coordonnées et POI récupérés ───────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    quartiers_test = list(_NOMINATIM_QUERIES.keys())
    gps, poi = build_geo_tables(quartiers_test)

    print("\n=== Coordonnées GPS (Nominatim) ===")
    for q, (lat, lon) in gps.items():
        fb = _GPS_FALLBACK.get(q, (0, 0))
        delta_lat = abs(lat - fb[0])
        delta_lon = abs(lon - fb[1])
        print(f"  {q:18s} : lat={lat:.5f}  lon={lon:.5f}  "
              f"(Δlat={delta_lat:.4f}, Δlon={delta_lon:.4f})")

    print("\n=== POI par quartier (Overpass, rayon 1 km) ===")
    print(f"  {'Quartier':18s} | {'Écoles':>7} | {'Mosquées':>8} | {'Commerces':>9} | {'Hôpitaux':>8} | {'Total':>6}")
    print("  " + "-" * 70)
    for q, counts in poi.items():
        total = sum(counts.values())
        print(f"  {q:18s} | {counts['ecoles']:>7} | {counts['mosquees']:>8} | "
              f"{counts['commerces']:>9} | {counts['hopitaux']:>8} | {total:>6}")
