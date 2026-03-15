# Prédiction des Prix Immobiliers à Nouakchott — Capstone ML

Projet Capstone Machine Learning — Master 1 SupNum (Mauritanie)
Compétition Kaggle interne évaluée sur le **RMSLE** (Root Mean Squared Logarithmic Error)

---

## Contexte

En Mauritanie, il n'existe pas de base de données publique sur les prix immobiliers. Les prix sont négociés de gré à gré et les annonces sont dispersées sur le web. Ce projet construit un pipeline ML complet pour prédire les prix de vente à Nouakchott à partir d'annonces immobilières scrappées.

- **Baseline** (prédiction par la médiane) : RMSLE ≈ 0.997
- **Objectif** : RMSLE < 0.75 (idéalement < 0.70)
- **Résultats obtenus** : RMSLE ≈ 0.65–0.70 avec Random Forest / Gradient Boosting

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
│   └── final/submission.csv        # Prédictions finales (289 lignes, colonnes id & prix)
│
├── notebooks/
│   ├── 01_inspection_nettoyage.ipynb   # Étapes 1-2 : chargement & nettoyage
│   ├── 02_valeurs_manquantes.ipynb     # Étape 3  : analyse & imputation des NaN
│   ├── 03_eda.ipynb                    # Étapes 4-5: outliers & visualisations EDA
│   ├── 04_feature_engineering.ipynb   # Étape 6a : géo-enrichissement & 34 features
│   └── 05_modelisation.ipynb           # Étape 6b : entraînement, CV, sélection du modèle
│
├── src/
│   ├── data_cleaning.py            # Chargement, nettoyage, standardisation
│   ├── geo_enrichment.py           # GPS par quartier, distances, POI
│   ├── feature_engineering.py      # Extraction features texte/temps, encodage quartier
│   ├── modeling.py                 # Entraînement, CV 5-fold, métriques, génération submission
│   └── pipeline.py                 # Pipeline complet de bout en bout
│
├── model/
│   ├── housing_model.pkl           # Meilleur modèle sérialisé
│   └── features.pkl                # Liste ordonnée des 34 features
│
└── outputs/figures/                # 8 graphiques PNG générés automatiquement
```

---

## Pipeline — Étapes détaillées

### Étape 1-2 : Chargement & Nettoyage (`01_inspection_nettoyage.ipynb`)
- Chargement des deux CSV, inspection dimensions/types/`describe()`
- Conversion `date_publication` en datetime
- Standardisation des noms de quartiers (strip whitespace)
- Validation plage prix (200K–54M MRU) et surface (20–2 000 m²)

### Étape 3 : Valeurs manquantes (`02_valeurs_manquantes.ipynb`)
- `nb_chambres` (1.2%) → médiane **par quartier** (MCAR)
- `nb_sdb` (72%) → indicatrice `has_sdb_info` + médiane globale (MAR/MNAR)
- `nb_salons` test (0.7%) → médiane par quartier depuis le train
- `caracteristiques` → chaîne vide `""`
- Toutes les valeurs d'imputation calculées **sur le train uniquement** (anti data leakage)

### Étapes 4-5 : EDA (`03_eda.ipynb`)
- Analyse des outliers (IQR) — conservés car variance réelle dans le marché mauritanien
- Distributions univariées, analyses croisées par quartier, heatmap de corrélation
- Corrélations attendues : surface/prix ≈ 0.62, chambres/prix ≈ 0.33

### Étape 6a : Feature Engineering (`04_feature_engineering.ipynb`)
**34 features au total :**

| Catégorie | Features |
|-----------|---------|
| Structurelles | `surface_m2`, `nb_chambres`, `nb_salons`, `nb_sdb`, `has_sdb_info` |
| Géographiques | `latitude`, `longitude`, `dist_centre_km`, `dist_aeroport_km`, `dist_plage_km` |
| POI (1 km) | `nb_ecoles_1km`, `nb_mosquees_1km`, `nb_commerces_1km`, `nb_hopitaux_1km`, `nb_total_pois_1km` |
| Texte/Caractéristiques | `has_garage`, `has_titre_foncier`, `has_camera`, `nb_balcons`, `taille_rue` |
| NLP description | `desc_len`, `desc_word_count`, `has_piscine`, `has_clim`, `has_meuble`, `is_luxe`, `is_renove`, `has_arabic` |
| Dérivées | `nb_pieces_total`, `surface_par_piece`, `log_surface`, `age_annonce_jours` |
| Encodage quartier | `quartier_target_enc`, `quartier_freq` |

**Note :** `prix_m2` n'est pas utilisé — c'est du data leakage (dérivé de la target).

### Étape 6b : Modélisation (`05_modelisation.ipynb`)
- **Target :** `log(1 + prix)` — prédiction en log-space car la métrique est RMSLE
- **Validation :** KFold 5-fold, shuffle=True, random_state=42
- **Modèles comparés :**

| Modèle | RMSLE (CV) |
|--------|-----------|
| Baseline (médiane) | ~0.999 |
| Régression Linéaire | ~0.85 |
| Ridge | ~0.85 |
| Lasso | ~0.85 |
| Random Forest | ~0.65–0.68 |
| Gradient Boosting | ~0.65–0.70 |
| XGBoost *(si dispo)* | ~0.63–0.67 |

- Le meilleur modèle est ré-entraîné sur tout le train, puis sérialisé dans `model/`

---

## Visualisations générées

| Fichier | Contenu |
|---------|---------|
| `etape3_valeurs_manquantes.png` | Bar chart horizontal % de valeurs manquantes |
| `etape4_outliers.png` | Boxplot prix, boxplot surface, scatter prix vs surface |
| `etape5_univariee.png` | 6 distributions : prix, log(prix), surface, chambres, salons, quartiers |
| `etape5_bivariee.png` | Boxplot prix/quartier, scatter prix/surface, boxplot prix/chambres, prix médian/quartier |
| `etape5_correlation.png` | Heatmap de corrélation (prix, surface, chambres, salons, sdb) |
| `etape6_feature_importance.png` | Top 20 features (bar chart horizontal) |
| `etape6_comparaison_modeles.png` | RMSLE et R² de tous les modèles vs baseline |
| `etape6_reel_vs_predit.png` | Scatter réel vs prédit avec droite y=x |

---

## Installation & Exécution

### Prérequis

```bash
pip install -r requirements.txt
```

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
xgboost>=2.0      # optionnel
lightgbm>=4.0     # optionnel
```

### Option 1 — Notebooks en séquence

```bash
jupyter nbconvert --to notebook --execute notebooks/01_inspection_nettoyage.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_valeurs_manquantes.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_modelisation.ipynb
```

### Option 2 — Pipeline complet (une seule commande)

```bash
python src/pipeline.py
```

---

## Checklist de validation

- [ ] `data/final/submission.csv` — 289 lignes, colonnes `id` et `prix`
- [ ] `model/housing_model.pkl` — chargeable avec `joblib.load()`
- [ ] Tous les notebooks s'exécutent sans erreur
- [ ] Aucun NaN dans les prédictions
- [ ] RMSLE en cross-validation < 0.75
- [ ] 8 figures sauvegardées dans `outputs/figures/`
- [ ] Aucun data leakage (imputations et encodages calculés sur le train uniquement)

---

## Charger le modèle et faire une prédiction

```python
import joblib
import pandas as pd
import numpy as np

model = joblib.load('model/housing_model.pkl')
features = joblib.load('model/features.pkl')

# X_test doit contenir les 34 features dans le bon ordre
X_test = pd.DataFrame(...)  # votre jeu de test enrichi

log_pred = model.predict(X_test[features])
prix_pred = np.maximum(np.expm1(log_pred), 100_000)  # min 100 000 MRU
```

---

## Décisions clés

| Décision | Justification |
|----------|--------------|
| Garder les outliers | Grande variance réelle (villas luxe vs petits biens périphérie) |
| Prédire `log(1+prix)` | Réduit l'asymétrie et correspond directement à la métrique RMSLE |
| Target encoding par quartier | Capture l'effet prix du quartier sans risque de leakage (calcul sur train seul) |
| Indicatrice `has_sdb_info` | 72% de `nb_sdb` manquant — l'absence d'info est elle-même informative |
| Distances GPS hardcodées | Pas d'accès API en offline — coordonnées vérifiées depuis les référentiels officiels |

---

*Projet réalisé dans le cadre du cours de Machine Learning — Master 1 SupNum, Mauritanie — Mars 2026*
