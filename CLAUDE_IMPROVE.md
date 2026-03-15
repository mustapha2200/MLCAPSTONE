# CLAUDE.md — Améliorer le RMSLE de 0.624 → 0.56

## Situation actuelle

- Mon score actuel sur le leaderboard Kaggle : **RMSLE = 0.62405** (6ème place)
- Le top 1 a : **RMSLE = 0.55941**
- Écart à combler : **~0.065 points de RMSLE**
- Modèle actuel : Random Forest avec 34 features, CV RMSLE = 0.6576
- Maximum 5 soumissions par jour

## Les données

- `data/raw/kaggle_train.csv` — 1153 annonces avec `prix` (target)
- `data/raw/kaggle_test.csv` — 289 annonces sans `prix`
- Colonnes : id, titre, prix, surface_m2, nb_chambres, nb_salons, nb_sdb, quartier, description, caracteristiques, source, date_publication
- 8 quartiers de Nouakchott, prix en MRU
- `nb_sdb` manquant à ~72%, `caracteristiques` à ~13.6%
- `description` en français et/ou arabe — mine d'or inexploitée
- Métrique : RMSLE (on prédit log1p(prix), on reconvertit avec expm1)

## Ce que j'ai déjà

Le pipeline actuel est dans `src/pipeline.py`. Les features actuelles (34) :
- Numériques : surface_m2, nb_chambres, nb_salons, nb_sdb, has_sdb_info
- Géo : latitude, longitude, dist_centre_km, dist_aeroport_km, dist_plage_km
- POI : nb_ecoles_1km, nb_mosquees_1km, nb_commerces_1km, nb_hopitaux_1km, nb_total_pois_1km
- Caractéristiques : has_garage, has_titre_foncier, has_camera, nb_balcons, taille_rue
- NLP basique : desc_len, desc_word_count, has_piscine, has_clim, has_meuble, is_luxe, is_renove, has_arabic
- Dérivées : nb_pieces_total, surface_par_piece, log_surface, age_annonce_jours
- Encodage : quartier_target_enc, quartier_freq

## Ta mission

Améliorer mon score RMSLE de 0.624 à moins de 0.57. Crée un nouveau script `src/improve_model.py` qui :
1. Charge les données
2. Applique TOUTES les améliorations ci-dessous
3. Compare les scores en CV 5-fold
4. Génère le meilleur `submission.csv`

## AMÉLIORATIONS À IMPLÉMENTER — PAR PRIORITÉ

### PRIORITÉ 1 : Feature Engineering avancé (plus gros impact attendu)

#### A) NLP avancé sur `description` et `titre`

Le texte est une mine d'or que je sous-exploite. Implémenter :

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# 1. TF-IDF sur description (français + arabe)
# Combiner titre + description
train['text_combined'] = train['titre'].fillna('') + ' ' + train['description'].fillna('')
test['text_combined'] = test['titre'].fillna('') + ' ' + test['description'].fillna('')

# TF-IDF avec top 100-200 termes
tfidf = TfidfVectorizer(max_features=200, min_df=3, max_df=0.95, ngram_range=(1, 2))
tfidf_train = tfidf.fit_transform(train['text_combined'])
tfidf_test = tfidf.transform(test['text_combined'])

# Réduction dimensionnelle avec SVD (garder 15-30 composantes)
svd = TruncatedSVD(n_components=20, random_state=42)
tfidf_train_svd = svd.fit_transform(tfidf_train)
tfidf_test_svd = svd.transform(tfidf_test)

# Ajouter comme features : tfidf_0, tfidf_1, ..., tfidf_19
for i in range(20):
    train[f'tfidf_{i}'] = tfidf_train_svd[:, i]
    test[f'tfidf_{i}'] = tfidf_test_svd[:, i]
```

#### B) Mots-clés spécifiques dans le texte arabe et français

Extraire des mots-clés indicateurs de prix depuis les descriptions. Chercher en **arabe ET français** :

```python
# Mots français indicateurs de standing
keywords_fr = {
    'has_etage': r'étage|etage|طابق|طوابق',          # multi-étages = plus cher
    'has_duplex': r'duplex|دوبلكس',
    'has_jardin': r'jardin|حديقة',
    'has_terrasse': r'terrasse|تراس',
    'has_cuisine_equip': r'cuisine équipée|مطبخ مجهز',
    'is_neuf': r'neuf|جديد|nouveau',
    'has_carrelage': r'carrelage|كارو|caro',
    'has_2_facades': r'façade|واجه',
    'mention_coin': r'coin|ركن|angle',                 # coin de rue = plus cher
    'mention_route': r'route|شارع|boulevard|طريق',    # sur une grande route
    'mention_prix_neg': r'négociable|قابل للتفاوض',
    'mention_urgent': r'urgent|مستعجل|فرصة',          # urgence = prix bas possible
    'nb_etages': None,  # extraire le nombre via regex
}

# Compter les mentions de chiffres dans la description (prix mentionné = signal)
train['desc_has_numbers'] = train['description'].str.contains(r'\d{6,}', na=False, regex=True).astype(int)
train['desc_nb_digits'] = train['description'].str.count(r'\d').fillna(0)
```

#### C) Features d'interaction

```python
# Interactions surface × quartier
train['surface_x_target_enc'] = train['surface_m2'] * train['quartier_target_enc']

# Chambres par rapport à la norme du quartier
quartier_chambre_median = train.groupby('quartier')['nb_chambres'].median()
train['chambres_vs_quartier'] = train['nb_chambres'] - train['quartier'].map(quartier_chambre_median)

# Surface par rapport à la norme du quartier
quartier_surface_median = train.groupby('quartier')['surface_m2'].median()
train['surface_vs_quartier'] = train['surface_m2'] - train['quartier'].map(quartier_surface_median)

# Ratio chambres/surface (densité de pièces)
train['densite_pieces'] = train['nb_pieces_total'] / train['surface_m2'].clip(lower=20)

# Log nb_chambres
train['log_chambres'] = np.log1p(train['nb_chambres'])

# Surface²  (non-linéarité)
train['surface_squared'] = train['surface_m2'] ** 2
```

#### D) Features temporelles avancées

```python
train['mois_publication'] = train['date_publication'].dt.month
train['jour_semaine'] = train['date_publication'].dt.dayofweek
train['is_weekend'] = (train['jour_semaine'] >= 5).astype(int)
train['trimestre'] = train['date_publication'].dt.quarter
```

#### E) Meilleur encodage du quartier

```python
# Target encoding avec régularisation (smoothing)
# Évite le surajustement sur les quartiers avec peu d'annonces (Riyadh=13, Sebkha=10)
global_mean = train['prix'].mean()
smoothing = 30  # paramètre de régularisation

quartier_stats = train.groupby('quartier')['prix'].agg(['mean', 'count'])
quartier_stats['smoothed_target'] = (
    (quartier_stats['count'] * quartier_stats['mean'] + smoothing * global_mean) /
    (quartier_stats['count'] + smoothing)
)
train['quartier_target_smooth'] = train['quartier'].map(quartier_stats['smoothed_target'])

# Aussi encoder la variance du prix par quartier (mesure l'homogénéité)
quartier_std = train.groupby('quartier')['prix'].std()
train['quartier_prix_std'] = train['quartier'].map(quartier_std)
```

### PRIORITÉ 2 : Modèles plus puissants

#### A) XGBoost optimisé

```python
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,        # L1
    reg_lambda=1.0,       # L2
    min_child_weight=5,
    gamma=0.1,
    random_state=42,
    early_stopping_rounds=50,
    eval_metric='rmse',
)

# Entraîner avec early stopping sur un fold de validation
```

#### B) LightGBM

```python
import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_samples=10,
    num_leaves=31,
    random_state=42,
)
```

#### C) CatBoost (gère les catégorielles nativement)

```python
from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=100,
    cat_features=['quartier'],  # passer le quartier en brut !
)
```

### PRIORITÉ 3 : Stacking / Blending (souvent le différenciateur du top 1-3)

```python
from sklearn.model_selection import KFold
import numpy as np

def get_oof_predictions(model, X, y, X_test, n_folds=5):
    """Out-of-fold predictions pour le stacking."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_train = np.zeros(len(X))
    oof_test = np.zeros(len(X_test))
    oof_test_folds = np.zeros((len(X_test), n_folds))

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        oof_train[val_idx] = model.predict(X_val)
        oof_test_folds[:, i] = model.predict(X_test)

    oof_test = oof_test_folds.mean(axis=1)
    return oof_train, oof_test

# Niveau 1 : obtenir les OOF de chaque modèle
oof_rf, test_rf = get_oof_predictions(rf_model, X_train, y_train, X_test)
oof_xgb, test_xgb = get_oof_predictions(xgb_model, X_train, y_train, X_test)
oof_lgb, test_lgb = get_oof_predictions(lgb_model, X_train, y_train, X_test)
oof_ridge, test_ridge = get_oof_predictions(ridge_model, X_train, y_train, X_test)

# Niveau 2 : méta-modèle (Ridge simple)
stack_train = pd.DataFrame({'rf': oof_rf, 'xgb': oof_xgb, 'lgb': oof_lgb, 'ridge': oof_ridge})
stack_test = pd.DataFrame({'rf': test_rf, 'xgb': test_xgb, 'lgb': test_lgb, 'ridge': test_ridge})

meta_model = Ridge(alpha=1.0)
meta_model.fit(stack_train, y_train)
final_pred = meta_model.predict(stack_test)
```

**Alternative simple — Blending par moyenne pondérée :**
```python
# Pondérer selon la performance CV de chaque modèle
# Plus le RMSLE est bas, plus le poids est élevé
weights = {'rf': 0.25, 'xgb': 0.35, 'lgb': 0.30, 'ridge': 0.10}
final_pred = (
    weights['rf'] * test_rf +
    weights['xgb'] * test_xgb +
    weights['lgb'] * test_lgb +
    weights['ridge'] * test_ridge
)
```

### PRIORITÉ 4 : Hyperparameter tuning

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist_rf = {
    'n_estimators': [200, 300, 500, 800],
    'max_depth': [8, 10, 12, 15, 20, None],
    'min_samples_leaf': [2, 3, 5, 8],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.5, 0.7],
}

# Scorer RMSLE négatif (sklearn maximise)
from sklearn.metrics import make_scorer
def neg_rmsle(y_true, y_pred):
    return -np.sqrt(np.mean((y_true - y_pred)**2))

rmsle_scorer = make_scorer(neg_rmsle)

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_dist_rf,
    n_iter=50,
    cv=5,
    scoring=rmsle_scorer,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)
print(f"Meilleurs params : {search.best_params_}")
print(f"Meilleur score : {-search.best_score_:.4f}")
```

### PRIORITÉ 5 : Nettoyage des outliers (prudent)

Plutôt que supprimer les outliers, essayer le **winsorizing** :

```python
# Clipper les prix extrêmes au P1 et P99
p1, p99 = train['prix'].quantile(0.01), train['prix'].quantile(0.99)
train['prix_clipped'] = train['prix'].clip(p1, p99)
y_train = np.log1p(train['prix_clipped'])

# Ou essayer de retirer les 10-20 annonces les plus aberrantes
# (surface 20m² avec prix > 10M = probable erreur)
mask = ~((train['surface_m2'] < 50) & (train['prix'] > 10_000_000))
train_clean = train[mask]
```

### PRIORITÉ 6 : Post-processing des prédictions

```python
# Clipper les prédictions extrêmes
test_pred = np.clip(test_pred, train['prix'].quantile(0.01), train['prix'].quantile(0.99))

# Ou ajuster par quartier si le modèle sous/sur-estime systématiquement
# Calibration par quartier sur le train (fold-based pour éviter le leakage)
```

---

## Structure du script à créer

Crée `src/improve_model.py` avec cette structure :

```python
#!/usr/bin/env python3
"""Script d'amélioration du modèle — objectif RMSLE < 0.57"""

# 1. Charger les données
# 2. Nettoyage (comme avant)
# 3. Feature engineering AMÉLIORÉ (toutes les priorités ci-dessus)
# 4. Comparer les modèles en CV 5-fold :
#    - Random Forest tuné
#    - XGBoost
#    - LightGBM
#    - CatBoost (si disponible)
#    - Ridge
# 5. Stacking des meilleurs modèles
# 6. Afficher tableau comparatif
# 7. Générer submission.csv avec le meilleur
# 8. Sauvegarder le modèle

# IMPORTANT : afficher à chaque étape le RMSLE CV pour voir l'impact
```

## Règles

- **PAS de data leakage** — target encoding, TF-IDF, etc. fit sur train only
- **random_state=42** partout
- Target = `np.log1p(prix)`, prédictions finales = `np.expm1(pred)`
- Minimum 100,000 MRU pour toute prédiction
- Submission = 289 lignes, colonnes `id,prix`
- **Afficher le RMSLE CV à chaque étape** pour mesurer l'impact de chaque amélioration
- Maximum 5 soumissions/jour → tester en CV d'abord

## Objectif chiffré

| Étape | RMSLE attendu |
|-------|--------------|
| Modèle actuel | 0.6576 |
| + features avancées | ~0.63 |
| + XGBoost/LightGBM | ~0.60 |
| + stacking | ~0.58 |
| + tuning fin | ~0.56 |

Chaque amélioration doit être mesurée indépendamment en CV.
