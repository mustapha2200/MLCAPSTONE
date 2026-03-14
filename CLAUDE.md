# CLAUDE.md — Projet Capstone : Prédiction des Prix Immobiliers en Mauritanie

## Qui tu es

Tu es un Data Scientist senior qui implémente un projet Capstone ML de bout en bout pour des étudiants Master 1 à SupNum (Mauritanie). Tu produis du code Python propre, documenté, reproductible, prêt à être évalué par un professeur.

## Contexte du projet

En Mauritanie, il n'existe pas de base de données publique sur les prix immobiliers. Les prix sont négociés de gré à gré et les annonces sont dispersées. Le but est de construire un outil de prédiction des prix immobiliers à Nouakchott.

C'est une compétition Kaggle interne. L'évaluation est sur le **RMSLE** (Root Mean Squared Logarithmic Error). La baseline (prédiction par la médiane) donne RMSLE ≈ 0.997. L'objectif est de faire significativement mieux.

## Les données

Deux fichiers CSV sont fournis dans `data/` :

- `kaggle_train.csv` — 1 153 annonces avec la variable cible `prix`
- `kaggle_test.csv` — 289 annonces sans `prix`

### Colonnes

| Colonne | Type | Description |
|---------|------|-------------|
| `id` | int | Identifiant unique |
| `titre` | str | Titre de l'annonce (français ou arabe) |
| `prix` | float | **Variable cible** — Prix en MRU (1 EUR ≈ 400 MRU). Train uniquement |
| `surface_m2` | float | Surface en m² |
| `nb_chambres` | float | Nombre de chambres (~1.2% manquant) |
| `nb_salons` | float | Nombre de salons |
| `nb_sdb` | float | Nombre de salles de bain (**~72% manquant**) |
| `quartier` | str | Quartier de Nouakchott (8 quartiers) |
| `description` | str | Description libre (français, arabe, ou mixte) |
| `caracteristiques` | str | Caractéristiques listées (~13.6% manquant) |
| `source` | str | Site web source (tout voursa.com) |
| `date_publication` | str | Date de publication |

### Les 8 quartiers (classés par prix médian décroissant)

| Quartier | Prix médian | GPS (lat, lon) | Caractère |
|----------|------------|----------------|-----------|
| Tevragh Zeina | ~6.5M MRU | 18.1036, -15.9785 | Huppé, ambassades, villas |
| Sebkha | ~3.9M MRU | 18.0730, -15.9870 | Commercial, marchés |
| Ksar | ~2.9M MRU | 18.0866, -15.9750 | Centre historique |
| Teyarett | ~2.9M MRU | 18.0950, -15.9700 | Centre, mixte |
| Dar Naim | ~1.7M MRU | 18.1200, -15.9450 | Résidentiel, expansion |
| Arafat | ~1.3M MRU | 18.0550, -15.9610 | Populaire, dense |
| Toujounine | ~1.1M MRU | 18.0680, -15.9350 | Périphérie, récent |
| Riyadh | ~0.85M MRU | 18.0850, -15.9550 | Résidentiel moyen (13 annonces seulement) |

### Points de référence GPS

- Centre-ville (Ksar) : 18.0866, -15.9750
- Aéroport Oumtounsy : 18.0987, -15.9476
- Plage (El Mina) : 18.0580, -16.0200

## Structure du repository à produire

```
mauritania-housing-ml/
├── CLAUDE.md                          ← ce fichier
├── README.md                          ← description du projet
├── requirements.txt                   ← dépendances Python
├── data/
│   ├── raw/
│   │   ├── kaggle_train.csv
│   │   └── kaggle_test.csv
│   ├── processed/
│   │   └── enriched_data.csv          ← après geo-enrichissement
│   └── final/
│       └── submission.csv             ← prédictions finales
├── notebooks/
│   ├── 01_inspection_nettoyage.ipynb   ← Étapes 1-2
│   ├── 02_valeurs_manquantes.ipynb     ← Étape 3
│   ├── 03_eda.ipynb                    ← Étapes 4-5
│   ├── 04_feature_engineering.ipynb    ← Étape 6 partie 1
│   └── 05_modelisation.ipynb           ← Étape 6 partie 2
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py               ← fonctions de nettoyage
│   ├── feature_engineering.py          ← fonctions de feature engineering
│   ├── geo_enrichment.py              ← géocodage + distances + POI
│   └── modeling.py                     ← entraînement + évaluation
├── model/
│   ├── housing_model.pkl              ← meilleur modèle sérialisé
│   ├── scaler.pkl                     ← scaler si utilisé
│   └── features.pkl                   ← liste des features
└── outputs/
    └── figures/                        ← tous les graphiques EDA
```

## Ce que tu dois implémenter — étape par étape

### ÉTAPE 1 : Chargement & Inspection (`notebooks/01_inspection_nettoyage.ipynb`)

- Charger `kaggle_train.csv` et `kaggle_test.csv`
- Afficher les dimensions, types, describe(), head()
- Vérifier que `prix` est absent du test
- Lister les quartiers et leur fréquence
- Afficher la plage de dates et les sources
- Identifier les colonnes numériques vs catégorielles

### ÉTAPE 2 : Nettoyage (`notebooks/01_inspection_nettoyage.ipynb`)

- Convertir `date_publication` en datetime
- Vérifier que tous les prix sont en MRU (range : 200K → 54M)
- Standardiser les noms de quartiers (strip whitespace)
- Confirmer : dataset = ventes résidentielles uniquement, pas de locations
- Vérifier la cohérence surface (20 → 2000 m²)

### ÉTAPE 3 : Valeurs manquantes (`notebooks/02_valeurs_manquantes.ipynb`)

**Analyser les mécanismes :**
- `nb_chambres` (1.2% manquant) → **MCAR** (manquant aléatoirement)
- `nb_sdb` (72.2% manquant) → **MAR/MNAR** (champ non rempli par les annonceurs)
- `caracteristiques` (13.6%) → **MAR** (rien de spécial à signaler)
- `nb_salons` dans le test (0.7%) → **MCAR**

**Stratégie d'imputation :**
- `nb_chambres` : médiane **par quartier**
- `nb_sdb` : créer une indicatrice `has_sdb_info` (0/1) + imputer par médiane globale du train
- `nb_salons` (test) : médiane par quartier depuis le train
- `caracteristiques` : NaN → chaîne vide `""`

**Important :** toujours calculer les valeurs d'imputation sur le **train uniquement** pour éviter le data leakage.

**Visualisation :** bar chart horizontal des % de valeurs manquantes avant imputation.

### ÉTAPE 4 : Outliers (`notebooks/03_eda.ipynb`)

- Calculer les bornes IQR pour `prix` et `surface_m2`
- Prix : ~54 outliers (4.7%) au-dessus de 12.5M MRU
- Surface : ~93 outliers (8.1%) au-dessus de 525 m²
- **Décision : NE PAS supprimer les outliers.** Dans le contexte mauritanien, la grande variance est réelle (villas de luxe Tevragh Zeina vs petits biens périphérie). On utilisera `log(prix)` pour réduire l'asymétrie.
- **Visualisation :** 3 subplots — boxplot prix, boxplot surface, scatter prix vs surface

### ÉTAPE 5 : EDA Uni/Bi/Multivariée (`notebooks/03_eda.ipynb`)

**5.1 Analyse univariée (6 subplots) :**
- Histogramme du prix (+ ligne médiane en rouge)
- Histogramme de log(1+prix) — montrer que la distribution devient plus normale
- Histogramme surface_m2
- Bar plot nb_chambres
- Bar plot nb_salons
- Bar plot horizontal des quartiers (nombre d'annonces)

**5.2 Analyse bivariée (4 subplots) :**
- Boxplot prix par quartier (ordonné par médiane décroissante, ylim au P95)
- Scatter prix vs surface coloré par quartier (top 4 quartiers)
- Boxplot prix vs nb_chambres (filtrer ≤ 10)
- Bar horizontal prix médian par quartier (avec annotations en millions)

**5.3 Analyse multivariée :**
- Heatmap de corrélation (prix, surface, chambres, salons, sdb)
- Tableau statistiques par quartier (nb_annonces, prix_median, prix_moyen, surface_mediane, chambres_mediane)

**Corrélations attendues :** surface/prix ≈ 0.62, chambres/prix ≈ 0.33, salons/prix ≈ 0.27

### ÉTAPE 6 : Feature Engineering & Modélisation

#### 6.1 Géo-Enrichissement (`src/geo_enrichment.py` + `notebooks/04_feature_engineering.ipynb`)

Utiliser les coordonnées GPS hardcodées des quartiers (table ci-dessus) pour créer :

- `latitude`, `longitude` — depuis le mapping quartier → GPS
- `dist_centre_km` — distance haversine vers Ksar (18.0866, -15.9750)
- `dist_aeroport_km` — distance haversine vers l'aéroport (18.0987, -15.9476)
- `dist_plage_km` — distance haversine vers la plage (18.0580, -16.0200)

**Fonction haversine :**
```python
from math import radians, cos, sin, asin, sqrt
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))
```

**POI approximatifs par quartier** (proxy de densité urbaine, puisqu'on n'a pas accès à l'API Overpass en offline) :

```python
QUARTIER_POI = {
    'Tevragh Zeina': {'ecoles': 8, 'mosquees': 6, 'commerces': 15, 'hopitaux': 4},
    'Ksar':          {'ecoles': 6, 'mosquees': 10, 'commerces': 20, 'hopitaux': 3},
    'Teyarett':      {'ecoles': 5, 'mosquees': 8, 'commerces': 12, 'hopitaux': 2},
    'Sebkha':        {'ecoles': 4, 'mosquees': 7, 'commerces': 25, 'hopitaux': 2},
    'Arafat':        {'ecoles': 6, 'mosquees': 12, 'commerces': 10, 'hopitaux': 1},
    'Dar Naim':      {'ecoles': 4, 'mosquees': 5, 'commerces': 8, 'hopitaux': 1},
    'Riyadh':        {'ecoles': 3, 'mosquees': 4, 'commerces': 6, 'hopitaux': 1},
    'Toujounine':    {'ecoles': 3, 'mosquees': 6, 'commerces': 5, 'hopitaux': 0},
    'El Mina':       {'ecoles': 4, 'mosquees': 8, 'commerces': 8, 'hopitaux': 1},
}
```

Features POI : `nb_ecoles_1km`, `nb_mosquees_1km`, `nb_commerces_1km`, `nb_hopitaux_1km`, `nb_total_pois_1km`

#### 6.2 Feature Engineering (`src/feature_engineering.py` + `notebooks/04_feature_engineering.ipynb`)

**Depuis `caracteristiques` (regex sur le texte en minuscules) :**
- `has_garage` — contient "garage"
- `has_titre_foncier` — contient "titre foncier"
- `has_camera` — contient "cam"
- `nb_balcons` — extraire le nombre via regex `(\d+)\s*balcon`
- `taille_rue` — extraire via regex `taille rue:\s*([\d.]+)`, imputer médiane si absent

**Depuis `description` (NLP basique) :**
- `desc_len` — longueur en caractères
- `desc_word_count` — nombre de mots
- `has_piscine` — "piscine" dans description OU caractéristiques
- `has_clim` — "climatisation" ou "clim"
- `has_meuble` — "meubl"
- `is_luxe` — "luxe", "haut standing", "standing"
- `is_renove` — "rénov", "renov", "neuf", "nouveau"
- `has_arabic` — contient des caractères arabes (regex `[\u0600-\u06FF]`)

**Variables dérivées :**
- `nb_pieces_total` = nb_chambres + nb_salons
- `surface_par_piece` = surface_m2 / max(nb_pieces_total, 1)
- `log_surface` = log(1 + surface_m2)
- `age_annonce_jours` = (date_ref - date_publication).days — utiliser 2026-03-02 comme date de référence

**Encodage du quartier (ATTENTION AU DATA LEAKAGE) :**
- `quartier_target_enc` — moyenne du prix par quartier, calculée **uniquement sur le train**. Mapper sur le test.
- `quartier_freq` — fréquence relative du quartier dans le train. Mapper sur le test.

**⚠️ NE PAS utiliser `prix_m2 = prix / surface` comme feature — c'est du data leakage (dérivé de la target).**

**Target :** `log_prix = log(1 + prix)` — on prédit en log-space car la métrique est RMSLE.

#### 6.3 Liste finale des 34 features

```python
FEATURE_COLS = [
    'surface_m2', 'nb_chambres', 'nb_salons', 'nb_sdb', 'has_sdb_info',
    'latitude', 'longitude', 'dist_centre_km', 'dist_aeroport_km', 'dist_plage_km',
    'nb_ecoles_1km', 'nb_mosquees_1km', 'nb_commerces_1km', 'nb_hopitaux_1km', 'nb_total_pois_1km',
    'nb_pieces_total', 'has_garage', 'has_titre_foncier', 'has_camera', 'nb_balcons', 'taille_rue',
    'desc_len', 'desc_word_count',
    'has_piscine', 'has_clim', 'has_meuble', 'is_luxe', 'is_renove', 'has_arabic',
    'age_annonce_jours', 'surface_par_piece', 'log_surface',
    'quartier_target_enc', 'quartier_freq'
]
```

Après extraction, remplir tout NaN restant par la médiane de la colonne (train).

#### 6.4 Modélisation (`notebooks/05_modelisation.ipynb`)

**Target :** `y = log(1 + prix)` (np.log1p)

**Modèles à tester (minimum 5) :**

| Modèle | Sklearn | Hyperparamètres |
|--------|---------|-----------------|
| Régression Linéaire | `LinearRegression()` | défaut |
| Ridge | `Ridge(alpha=1.0)` | α=1.0 |
| Lasso | `Lasso(alpha=0.001)` | α=0.001 |
| Random Forest | `RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)` | |
| Gradient Boosting | `GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, min_samples_leaf=5, random_state=42)` | |

Si `xgboost` est disponible, ajouter `XGBRegressor` en bonus.

**Validation croisée :** KFold 5-fold, shuffle=True, random_state=42.

**Métriques à calculer par fold et moyenner :**
- RMSE (en log-space)
- MAE (en log-space)
- R²
- RMSLE (reconverti en échelle originale)

**Fonction RMSLE sur l'échelle originale :**
```python
def rmsle_original_scale(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))
```

**Baseline :** prédire la médiane du train pour tous → RMSLE ≈ 0.999.

**Résultats attendus :** le Random Forest ou Gradient Boosting devrait atteindre RMSLE ≈ 0.65-0.70, soit ~34% d'amélioration sur la baseline.

**Après sélection du meilleur modèle :**
1. Ré-entraîner sur tout le train
2. Afficher feature importances (top 20, bar chart horizontal)
3. Plot réel vs prédit (scatter avec ligne y=x en rouge)
4. Tableau comparatif de tous les modèles

#### 6.5 Génération de la soumission

```python
test_pred = np.maximum(np.expm1(model.predict(X_test)), 100000)
submission = pd.DataFrame({'id': test['id'], 'prix': test_pred.round(0).astype(int)})
submission.to_csv('data/final/submission.csv', index=False)
```

Le fichier doit contenir exactement **289 lignes** (hors header), colonnes `id` et `prix`.

#### 6.6 Sauvegarde du modèle

```python
import joblib
joblib.dump(best_model, 'model/housing_model.pkl')
joblib.dump(FEATURE_COLS, 'model/features.pkl')
```

## Visualisations attendues (figures/)

1. `etape3_valeurs_manquantes.png` — bar chart % manquant
2. `etape4_outliers.png` — 3 subplots (boxplot prix, boxplot surface, scatter)
3. `etape5_univariee.png` — 6 subplots distributions
4. `etape5_bivariee.png` — 4 subplots analyses croisées
5. `etape5_correlation.png` — heatmap corrélation
6. `etape6_feature_importance.png` — top 20 features bar chart
7. `etape6_comparaison_modeles.png` — 2 subplots (RMSLE + R²) avec baseline en rouge
8. `etape6_reel_vs_predit.png` — scatter réel vs prédit

**Style matplotlib :** `sns.set_style('whitegrid')`, figsize=(12,6) par défaut, dpi=150, police 12pt.

## Règles strictes

1. **PAS de data leakage** — le target encoding, les imputations, les médiane doivent être calculés sur le train UNIQUEMENT puis mappés sur le test
2. **PAS de `prix_m2` comme feature** — c'est dérivé de la target
3. **Prédire en log-space** (`log1p(prix)`) car la métrique est RMSLE
4. **Reconvertir avec `expm1`** pour les prédictions finales
5. **Assurer prix > 0** dans la soumission (minimum 100,000 MRU)
6. **Code reproductible** — random_state=42 partout
7. **Documenter chaque décision** — en markdown dans les notebooks
8. **Sauvegarder les données intermédiaires** — `data/processed/enriched_data.csv`

## requirements.txt

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
```

Optionnel : `xgboost>=2.0`, `lightgbm>=4.0`

## Commandes pour exécuter

```bash
# Installer les dépendances
pip install -r requirements.txt

# Exécuter les notebooks en séquence
jupyter nbconvert --to notebook --execute notebooks/01_inspection_nettoyage.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_valeurs_manquantes.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_modelisation.ipynb

# Ou exécuter le pipeline complet
python src/pipeline.py
```

## Comment vérifier que tout fonctionne

- [ ] `data/final/submission.csv` existe avec 289 lignes et colonnes `id,prix`
- [ ] `model/housing_model.pkl` existe et peut être chargé avec joblib
- [ ] Tous les notebooks s'exécutent sans erreur de A à Z
- [ ] Aucun NaN dans les prédictions
- [ ] RMSLE en cross-validation < 0.75 (idéalement < 0.70)
- [ ] Les figures sont sauvegardées dans `outputs/figures/`
- [ ] Le code ne contient aucune référence à des chemins locaux absolus
