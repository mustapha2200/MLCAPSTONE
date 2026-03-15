# Prédiction des Prix Immobiliers à Nouakchott — Capstone ML

Projet Capstone Machine Learning — Master 1 SupNum (Mauritanie)
Compétition Kaggle interne évaluée sur le **RMSLE** (Root Mean Squared Logarithmic Error)

---

## Résultats

| Version | Modèle | RMSLE CV | Score Kaggle |
|---------|--------|----------|-------------|
| Baseline | Médiane | ~0.999 | — |
| V1 | Random Forest (34 features) | 0.6576 | 0.62405 (6ème) |
| **V2** | **Stacking CatBoost+XGB+LGB+RF+Ridge (77 features)** | **0.6263** | En cours |
| Top 1 leaderboard | — | — | 0.55941 |

**Déploiement :**
- API : [https://mlcapstone.onrender.com](https://mlcapstone.onrender.com)
- Interface : [https://immobilier-nouakchott.vercel.app](https://immobilier-nouakchott.vercel.app)

---

## Contexte

En Mauritanie, il n'existe pas de base de données publique sur les prix immobiliers. Les prix sont négociés de gré à gré et les annonces sont dispersées sur le web. Ce projet construit un pipeline ML complet pour prédire les prix de vente à Nouakchott à partir d'annonces immobilières scrappées.

---

## Données

| Fichier | Lignes | Description |
|---------|--------|-------------|
| `data/raw/kaggle_train.csv` | 1 153 | Annonces avec la colonne `prix` (target) |
| `data/raw/kaggle_test.csv` | 289 | Annonces sans `prix` — à prédire |

### Colonnes principales

| Colonne | Type | Notes |
|---------|------|-------|
| `prix` | float | Prix en MRU (1 EUR ≈ 400 MRU). Train uniquement |
| `surface_m2` | float | Surface habitable |
| `nb_chambres` | float | ~1.2% manquant |
| `nb_sdb` | float | **~72% manquant** — indicatrice créée |
| `quartier` | str | 8 quartiers de Nouakchott |
| `description` | str | Texte libre (français/arabe) |
| `caracteristiques` | str | Liste de caractéristiques (~13.6% manquant) |

### Les 8 quartiers (prix médian décroissant)

| Quartier | Prix médian | Caractère |
|----------|------------|-----------|
| Tevragh Zeina | ~6.5M MRU | Huppé, ambassades, villas |
| Sebkha | ~3.9M MRU | Commercial, marchés |
| Ksar | ~2.9M MRU | Centre historique |
| Teyarett | ~2.9M MRU | Centre, mixte |
| Dar Naim | ~1.7M MRU | Résidentiel, expansion |
| Arafat | ~1.3M MRU | Populaire, dense |
| Toujounine | ~1.1M MRU | Périphérie |
| Riyadh | ~0.85M MRU | Résidentiel moyen |

---

## Structure du projet

```
mauritania-housing-ml/
├── data/
│   ├── raw/                        # Données Kaggle originales (ne pas modifier)
│   ├── processed/                  # Données intermédiaires (clean, imputed, enriched)
│   └── final/
│       ├── submission.csv          # Prédictions V1 (Random Forest)
│       └── submission_improved.csv # Prédictions V2 (Stacking — à soumettre)
│
├── notebooks/
│   ├── 01_inspection_nettoyage.ipynb   # Étapes 1-2 : chargement & nettoyage
│   ├── 02_valeurs_manquantes.ipynb     # Étape 3  : analyse & imputation des NaN
│   ├── 03_eda.ipynb                    # Étapes 4-5 : outliers & visualisations EDA
│   ├── 04_feature_engineering.ipynb   # Étape 6a : géo-enrichissement & features
│   └── 05_modelisation.ipynb           # Étape 6b : entraînement, CV, sélection
│
├── src/
│   ├── data_cleaning.py            # Chargement, nettoyage, standardisation
│   ├── geo_enrichment.py           # GPS par quartier, distances, POI
│   ├── feature_engineering.py      # Extraction features texte/temps, encodage quartier
│   ├── modeling.py                 # Entraînement, CV 5-fold, métriques
│   ├── pipeline.py                 # Pipeline V1 de bout en bout
│   └── improve_model.py            # Pipeline V2 amélioré (NLP + stacking)
│
├── api/
│   ├── app.py                      # API Flask (4 endpoints)
│   ├── predict.py                  # Reconstruction des features pour l'inférence
│   ├── config.py                   # GPS, POI, constantes
│   └── requirements.txt
│
├── frontend/                       # Application Next.js (TypeScript + Tailwind)
│   └── src/
│       ├── app/                    # Pages : accueil, prédiction, analyse
│       └── components/             # Carte Leaflet, formulaire, résultats
│
├── model/
│   ├── housing_model.pkl           # Modèle V1 (Random Forest)
│   ├── housing_model_improved.pkl  # Modèle V2 (CatBoost — meilleur individuel)
│   ├── features.pkl                # 34 features V1
│   └── features_improved.pkl       # 77 features V2
│
└── outputs/figures/                # 8 graphiques PNG générés automatiquement
```

---

## Pipeline V1 — Étapes détaillées

### Étape 1-2 : Chargement & Nettoyage
- Conversion `date_publication` en datetime, standardisation des quartiers
- Validation plage prix (200K–54M MRU) et surface (20–2 000 m²)

### Étape 3 : Valeurs manquantes
- `nb_chambres` (1.2%) → médiane par quartier | `nb_sdb` (72%) → indicatrice + médiane globale
- Toutes les imputations calculées **sur le train uniquement** (anti data leakage)

### Étapes 4-5 : EDA
- Outliers conservés (variance réelle du marché mauritanien)
- Corrélations : surface/prix ≈ 0.62, chambres/prix ≈ 0.33

### Étape 6a : Feature Engineering V1 (34 features)

| Catégorie | Features |
|-----------|---------|
| Structurelles | `surface_m2`, `nb_chambres`, `nb_salons`, `nb_sdb`, `has_sdb_info` |
| Géographiques | `latitude`, `longitude`, `dist_centre_km`, `dist_aeroport_km`, `dist_plage_km` |
| POI (1 km) | `nb_ecoles_1km`, `nb_mosquees_1km`, `nb_commerces_1km`, `nb_hopitaux_1km`, `nb_total_pois_1km` |
| Texte | `has_garage`, `has_titre_foncier`, `has_camera`, `nb_balcons`, `taille_rue` |
| NLP basique | `desc_len`, `desc_word_count`, `has_piscine`, `has_clim`, `has_meuble`, `is_luxe`, `is_renove`, `has_arabic` |
| Dérivées | `nb_pieces_total`, `surface_par_piece`, `log_surface`, `age_annonce_jours` |
| Encodage | `quartier_target_enc`, `quartier_freq` |

### Étape 6b : Modélisation V1

| Modèle | RMSLE CV |
|--------|----------|
| Baseline (médiane) | ~0.999 |
| Ridge / Lasso | ~0.85 |
| Random Forest | 0.658 |
| Gradient Boosting | 0.657 |

---

## Pipeline V2 — Améliorations (`src/improve_model.py`)

### Nouvelles features (77 au total)

**TF-IDF + SVD sur titre + description + caractéristiques :**
- TfidfVectorizer (300 termes, bigrammes) → TruncatedSVD (20 composantes)
- Exploite le texte arabe et français ensemble

**Mots-clés bilingues supplémentaires :**
- `has_etage`, `has_duplex`, `has_jardin`, `has_terrasse`, `has_carrelage`
- `mention_coin`, `mention_prix_neg`, `mention_urgent`, `has_2_facades`

**Features d'interaction :**
- `surface_x_target` — surface × prix moyen du quartier
- `chambres_vs_quartier` — écart à la médiane du quartier
- `surface_vs_quartier` — écart à la surface médiane du quartier
- `log_surface_x_target`, `densite_pieces`, `surface_squared`

**Encodage amélioré :**
- Target encoding avec **smoothing** (évite le surajustement sur Riyadh=13 et Sebkha=10 annonces)
- `quartier_prix_std` — variance du prix par quartier

**Temporel :**
- `mois_publication`, `trimestre`, `is_weekend`

**Winsorizing :**
- Clip de la target aux percentiles P1–P99 avant entraînement

### Résultats V2

| Modèle | RMSLE CV |
|--------|----------|
| Baseline | ~0.999 |
| **V1 Random Forest (34 features)** | **0.6576** |
| CatBoost (77 features) | 0.6330 |
| Random Forest (77 features) | 0.6390 |
| Ridge (77 features) | 0.6436 |
| XGBoost | 0.6475 |
| Blending pondéré | 0.6344 |
| **Stacking Ridge méta** | **0.6263** ← meilleur |

**Amélioration : −0.031 RMSLE** par rapport à V1

---

## Déploiement (Phase 5)

### API Flask — `https://mlcapstone.onrender.com`

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/health` | GET | Statut de l'API |
| `/api/predict` | POST | Estimation du prix d'un bien |
| `/api/stats` | GET | Statistiques du marché |
| `/api/quartiers` | GET | Liste des quartiers avec GPS |

### Frontend Next.js — `https://immobilier-nouakchott.vercel.app`

- Page d'accueil : carte Leaflet + statistiques marché + bar charts
- Page prédiction : formulaire → prix estimé + intervalle de confiance (P10–P90)
- Page analyse : feature importance + comparaison modèles

---

## Installation & Exécution

```bash
pip install -r requirements.txt
# optionnel pour V2 :
pip install xgboost lightgbm catboost
```

```bash
# Pipeline V1
python src/pipeline.py

# Pipeline V2 amélioré
python src/improve_model.py

# API Flask
cd api && python app.py

# Frontend
cd frontend && npm run dev
```

---

## Checklist de validation

- [x] `data/final/submission.csv` — 289 lignes V1
- [x] `data/final/submission_improved.csv` — 289 lignes V2 (stacking)
- [x] `model/housing_model.pkl` — Random Forest V1
- [x] `model/housing_model_improved.pkl` — CatBoost V2
- [x] API Flask déployée sur Render
- [x] Frontend Next.js déployé sur Vercel
- [x] Aucun data leakage (TF-IDF, encodages, imputations fitté sur train uniquement)
- [x] RMSLE CV V2 = 0.6263 < objectif 0.70

---

## Décisions clés

| Décision | Justification |
|----------|--------------|
| Garder les outliers | Grande variance réelle (villas luxe vs petits biens périphérie) |
| Prédire `log(1+prix)` | Réduit l'asymétrie, correspond directement à la métrique RMSLE |
| Target encoding avec smoothing | Évite le surajustement sur quartiers sous-représentés (Riyadh=13, Sebkha=10) |
| TF-IDF + SVD | Le texte FR/AR contient des signaux forts (étage, jardin, titre foncier) |
| Stacking vs meilleur modèle seul | +0.007 RMSLE grâce à la diversité des erreurs |
| Winsorizing P1–P99 | Réduit l'impact des prix aberrants sans supprimer les données |

---

*Projet réalisé dans le cadre du cours de Machine Learning — Master 1 SupNum, Mauritanie — Mars 2026*
