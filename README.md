# Prédiction des Prix Immobiliers à Nouakchott — Capstone ML

Projet Capstone Machine Learning — Master 1 SupNum (Mauritanie)  
Compétition Kaggle interne évaluée sur le **RMSLE** (Root Mean Squared Logarithmic Error)

---

## Résultats

| Version | Modèle / Changement clé | RMSLE CV | Score Kaggle |
|---------|--------------------------|----------|-------------|
| Baseline | Médiane | ~0.999 | — |
| V1 | Random Forest (34 features) | 0.6576 | 0.62405 |
| V2 | Stacking (77 features, leaky CV) | 0.6263 | ~0.624 |
| V3 | Stacking amélioré *(CV fictif — bug meta-learner in-sample)* | ~~0.5502~~ | 0.6109 |
| V4 | Correction leakage meta-learner + CV honnête | ~0.608 | 0.6083 |
| V5 | NLP transductif + clipping global | 0.5681 | 0.6034 |
| V6b | Extraction de prix depuis le texte + tuning XGB | 0.5681 | **0.5624** |
| **V7b** | **Early stopping dans la boucle OOF** | **0.5670** | **0.5610** |
| Top 1 leaderboard | — | — | 0.52173 |

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
│   ├── processed/                  # Données intermédiaires + geo_meta.json
│   └── final/
│       ├── submission.csv          # Prédictions V1
│       ├── submission_improved.csv # Prédictions dernière version
│       └── submission_v7b.csv      # Prédictions V7b (meilleure soumission)
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
│   ├── improve_model.py            # Pipeline V2 (NLP + stacking)
│   └── improve_model_v7b.py        # Pipeline V7b — version la plus récente
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
│   ├── housing_model_improved.pkl  # Modèle actuel (meilleur individuel)
│   ├── features.pkl                # 34 features V1
│   └── features_improved.pkl       # 117 features V7b
│
└── outputs/figures/                # Graphiques PNG générés automatiquement
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
| NLP basique | `desc_len`, `desc_word_count`, `has_piscine`, `has_clim`, `has_meuble`, `is_luxe`, `is_renove` |
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

## Pipeline V7b — Version finale (`src/improve_model_v7b.py`)

Le pipeline final (117 features, stacking 6 modèles, pseudo-labeling) construit sur les correctifs progressifs des versions V3 à V7b.

### Features (117 au total)

**NLP dual-langue (transductif — fitté sur train+test) :**
- TF-IDF français (word bigrammes, 300 termes) → TruncatedSVD (25 composantes)
- TF-IDF arabe (char n-grams, 200 termes) → TruncatedSVD (25 composantes)
- Vocabulaire fitté sur train+test combinés (transductif — aucun label impliqué)

**Extraction structurée depuis le texte :**
- `desc_price_extracted` — prix explicite extrait du texte ("3 500 000 MRU", "2 millions")
- `has_extracted_price`, `desc_price_log` — flag et version log
- `n_floors`, `construction_year`, `has_phone`, `has_price_mention`

**Mots-clés bilingues (FR + AR) :**
- `has_garage`, `has_piscine`, `has_clim`, `has_meuble`, `is_luxe`, `is_renove`
- `has_etage`, `has_duplex`, `has_jardin`, `has_terrasse`, `has_carrelage`
- `mention_coin`, `mention_prix_neg`, `mention_urgent`, `has_2_facades`

**Features d'interaction :**
- `surface_x_target`, `log_surface_x_target` — surface × prix moyen du quartier
- `chambres_vs_quartier`, `surface_vs_quartier` — écarts aux médianes du quartier
- `densite_pieces`, `surface_squared`

**Encodage cible (smoothed, per-fold) :**
- `quartier_target_enc` / `quartier_target_smooth` — smoothing k=30
- Recomputed à l'intérieur de chaque fold CV pour éviter le leakage
- `quartier_prix_std`, `quartier_freq`

**Temporel :** `mois_publication`, `trimestre`, `is_weekend`, `age_annonce_jours`

### Architecture de stacking

```
6 modèles de base (5-fold OOF + early stopping) :
  LightGBM (tuné Optuna)
  CatBoost (tuné Optuna)
  XGBoost  (tuné Optuna)
  Random Forest
  ExtraTrees
  Ridge

        ↓ prédictions OOF (1153 × 6)

Meta-learner : RidgeCV (kf_meta interne — évaluation honnête)

        ↓ meilleur de : stack Ridge vs blend pondéré

Pseudo-labeling (2 rounds, weight=0.35) :
  → test predictions used as pseudo-labels
  → retrain sur train + test pseudo-labelisé
  → CV scoré uniquement sur les vraies lignes train
```

### Tuning des hyperparamètres (Optuna)

- 100 essais par modèle (LGB, CatBoost, XGB)
- **Splits Optuna séparés** (`kf_tune`, seed différent) des splits d'évaluation finale (`kf`)
- Early stopping à 50 rounds dans Optuna ET dans la boucle OOF de stacking

### Résultats V7b

| Modèle | RMSLE OOF |
|--------|-----------|
| Random Forest | 0.5792 |
| LightGBM (tuné) | 0.5796 |
| XGBoost (tuné) | 0.5817 |
| ExtraTrees | 0.5907 |
| CatBoost (tuné) | 0.5999 |
| Ridge | 0.6465 |
| **Stacking Ridge (initial)** | **0.5782** |
| **Pseudo-labeling Round 2** | **0.5670** ← soumis |

---

## Bugs critiques identifiés et corrigés

### Bug V3 — Meta-learner évalué en in-sample
Le meta-learner était entraîné sur les prédictions OOF puis scoré sur ces mêmes lignes.  
**Symptôme :** CV affiché 0.5502, LB réel 0.6109 (gap de 0.061).  
**Correction :** ajout d'une boucle CV interne (`kf_meta`) pour le meta-learner.

### Bug V4 — Leakage target encoding
Le target encoding du quartier était calculé sur le train complet avant le split CV.  
**Correction :** recomputation à l'intérieur de chaque fold, uniquement sur les lignes de train du fold.

### Bug V7 — Early stopping absent de la boucle OOF
Optuna utilisait `early_stopping(50)` mais la boucle OOF stacking tournait tous les `n_estimators` sans arrêt.  
**Symptôme :** LGB Optuna 0.5775 vs OOF 0.6032 (gap de 0.026 sur un seul modèle).  
**Correction :** `early_stopping_rounds` intégré dans le constructeur + `eval_set` passé dans `fit()`.

### Fausse piste — LOO encoding (V7 initial)
Leave-One-Out encoding remplaçait le smoothed encoding.  
**Symptôme :** LB dégradé de 0.562 → 0.674.  
**Cause :** Riyadh n'a que 13 annonces — variance LOO trop élevée, contamine les features d'interaction.  
**Correction :** retour au target encoding lissé k=30.

### Fausse piste — Extraction prix par nombres bruts (V6 initial)
Regex capturant tout entier de 7–9 chiffres comme prix potentiel.  
**Symptôme :** LB dégradé de 0.562 → 0.654.  
**Cause :** les numéros de téléphone mauritaniens font 8 chiffres.  
**Correction :** restriction aux patterns explicites uniquement ("X MRU", "X millions", "prix: XXXXXX").

---

## Calibration CV / Leaderboard

| Version | CV RMSLE | Kaggle LB | Gap | Fiable ? |
|---------|----------|-----------|-----|----------|
| V3 | ~~0.5502~~ | 0.6109 | 0.061 | ❌ in-sample meta |
| V4 | ~0.608 | 0.6083 | ~0.000 | ✅ |
| V5 | 0.5681 | 0.6034 | −0.005 | ✅ |
| V6b | 0.5681 | 0.5624 | −0.006 | ✅ |
| V7b | 0.5670 | 0.5610 | −0.006 | ✅ |

Un gap CV/LB stable à ~0.006 indique que le CV est honnête et que les améliorations locales se traduisent fidèlement en améliorations sur le leaderboard.

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
pip install xgboost lightgbm catboost optuna
```

```bash
# Pipeline V1
python src/pipeline.py

# Pipeline V7b (version finale)
python src/improve_model_v7b.py

# API Flask
cd api && python app.py

# Frontend
cd frontend && npm run dev
```

---

## Checklist de validation

- [x] `data/final/submission_v7b.csv` — 289 lignes, meilleure soumission (LB 0.5610)
- [x] `model/housing_model_improved.pkl` — modèle courant chargé par l'API
- [x] API Flask déployée sur Render
- [x] Frontend Next.js déployé sur Vercel
- [x] Aucun data leakage (TF-IDF transductif ✓, target encoding per-fold ✓, meta-learner kf_meta ✓)
- [x] Early stopping cohérent entre Optuna et boucle OOF
- [x] CV/LB gap stable à ~0.006 (CV fiable)
- [x] RMSLE V7b = 0.5610 — amélioration de 43% vs baseline naïve (0.999)

---

## Décisions clés

| Décision | Justification |
|----------|--------------|
| Garder les outliers | Grande variance réelle (villas luxe vs petits biens périphérie) |
| Prédire `log(1+prix)` | Réduit l'asymétrie, correspond directement à la métrique RMSLE |
| Target encoding smoothed k=30 (per-fold) | Évite le surajustement sur quartiers sous-représentés (Riyadh=13) et le leakage |
| NLP transductif (fitté sur train+test) | Améliore les poids IDF pour les termes arabes rares du test set |
| Extraction prix depuis le texte | Beaucoup d'annonces mauritaniennes écrivent le prix en clair — signal quasi-direct |
| Early stopping identique entre Optuna et stacking | Sans cette cohérence : gap de +0.026 RMSLE sur LGB seul |
| Splits Optuna séparés des splits d'évaluation | Évite que les hyperparamètres soient sur-adaptés aux folds d'évaluation |
| Pseudo-labeling (weight=0.35, 2 rounds) | +25% de données supplémentaires, gain constant de ~0.011 RMSLE |
| LOO encoding abandonné | Bruit trop élevé sur petits quartiers — smoothing bat la précision à cette échelle |

---

*Projet réalisé dans le cadre du cours de Machine Learning — Master 1 SupNum, Mauritanie — Mars 2026*