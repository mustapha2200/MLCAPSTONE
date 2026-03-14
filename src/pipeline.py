"""
pipeline.py — Pipeline complet d'un seul coup (alternative aux notebooks)
Projet Capstone : Prédiction des Prix Immobiliers à Nouakchott
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_cleaning import load_data, clean_dataframe
from src.geo_enrichment import enrich_geo
from src.feature_engineering import (
    ImputationConfig, QuartierEncoder, build_features,
    add_sdb_indicator, FEATURE_COLS
)
from src.modeling import (
    compare_all_models, train_final_model, save_model,
    generate_submission, rmsle_original_scale, get_models,
    baseline_rmsle
)
from sklearn.model_selection import KFold

sns.set_style('whitegrid')

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'kaggle_train.csv')
TEST_PATH   = os.path.join(BASE_DIR, 'data', 'raw', 'kaggle_test.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data', 'processed'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data', 'final'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'model'), exist_ok=True)


def step1_2_load_clean():
    print("\n" + "="*60)
    print("ÉTAPE 1-2 : Chargement et nettoyage")
    print("="*60)
    train_raw, test_raw = load_data(TRAIN_PATH, TEST_PATH)
    print(f"Train brut : {train_raw.shape}")
    print(f"Test brut  : {test_raw.shape}")

    train = clean_dataframe(train_raw, is_train=True)
    test  = clean_dataframe(test_raw,  is_train=False)
    print(f"Train nettoyé : {train.shape}")
    print(f"Test nettoyé  : {test.shape}")
    return train, test


def step3_missing(train, test):
    print("\n" + "="*60)
    print("ÉTAPE 3 : Valeurs manquantes")
    print("="*60)

    # Indicatrice nb_sdb AVANT imputation
    train = add_sdb_indicator(train)
    test  = add_sdb_indicator(test)

    imputer = ImputationConfig()
    imputer.fit(train)

    # Figure valeurs manquantes AVANT imputation
    cols_fig = ['nb_chambres', 'nb_sdb', 'nb_salons', 'caracteristiques', 'description']
    raw_pcts = {c: train[c].isna().mean() * 100 for c in cols_fig if c in train.columns}
    raw_pcts = dict(sorted({k: v for k, v in raw_pcts.items() if v > 0}.items(),
                            key=lambda x: x[1], reverse=True))

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(list(raw_pcts.keys()), list(raw_pcts.values()),
                   color='steelblue', edgecolor='white')
    for bar, val in zip(bars, raw_pcts.values()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10)
    ax.set_xlabel('% valeurs manquantes')
    ax.set_title('Valeurs manquantes par colonne — avant imputation (Train)', fontsize=13)
    ax.set_xlim(0, max(raw_pcts.values()) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'etape3_valeurs_manquantes.png'), dpi=150, bbox_inches='tight')
    plt.close()

    train = imputer.apply_basic(train)
    test  = imputer.apply_basic(test)

    print("Imputation terminée. NaN restants (train) :")
    for col in ['nb_chambres', 'nb_salons', 'nb_sdb']:
        print(f"  {col}: {train[col].isna().sum()}")

    return train, test, imputer


def step4_5_eda(train):
    print("\n" + "="*60)
    print("ÉTAPE 4-5 : EDA et Outliers")
    print("="*60)

    # Outliers
    def iqr_bounds(s):
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        return Q1 - 1.5*IQR, Q3 + 1.5*IQR

    lp, hp = iqr_bounds(train['prix'])
    ls, hs = iqr_bounds(train['surface_m2'])
    n_p = ((train['prix'] > hp) | (train['prix'] < lp)).sum()
    n_s = ((train['surface_m2'] > hs) | (train['surface_m2'] < ls)).sum()
    print(f"Outliers prix    : {n_p} ({n_p/len(train)*100:.1f}%)")
    print(f"Outliers surface : {n_s} ({n_s/len(train)*100:.1f}%)")
    print("Décision : Pas de suppression — log(prix) pour normaliser.")

    # Figure 4 : outliers
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].boxplot(train['prix']/1e6, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7))
    axes[0].set_title('Boxplot Prix (M MRU)')
    axes[1].boxplot(train['surface_m2'], patch_artist=True,
                    boxprops=dict(facecolor='darkorange', alpha=0.7))
    axes[1].set_title('Boxplot Surface (m²)')
    colors_q = plt.cm.tab10(np.linspace(0, 1, train['quartier'].nunique()))
    for i, (q, grp) in enumerate(train.groupby('quartier')):
        axes[2].scatter(grp['surface_m2'], grp['prix']/1e6, alpha=0.4, s=20,
                        label=q, color=colors_q[i])
    axes[2].set_xlabel('Surface (m²)')
    axes[2].set_ylabel('Prix (M MRU)')
    axes[2].set_title('Prix vs Surface')
    axes[2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'etape4_outliers.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 5 univariée
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0,0].hist(train['prix']/1e6, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0,0].axvline(train['prix'].median()/1e6, color='red', lw=2, label='Médiane')
    axes[0,0].set_title('Distribution Prix'); axes[0,0].legend()
    axes[0,1].hist(np.log1p(train['prix']), bins=40, color='teal', edgecolor='white', alpha=0.8)
    axes[0,1].set_title('Distribution log(1+prix)')
    axes[0,2].hist(train['surface_m2'], bins=40, color='darkorange', edgecolor='white', alpha=0.8)
    axes[0,2].set_title('Distribution Surface (m²)')
    c_counts = train['nb_chambres'].value_counts().sort_index()
    axes[1,0].bar(c_counts.index.astype(str), c_counts.values, color='mediumseagreen', edgecolor='white')
    axes[1,0].set_title('Répartition nb_chambres')
    s_counts = train['nb_salons'].value_counts().sort_index()
    axes[1,1].bar(s_counts.index.astype(str), s_counts.values, color='mediumpurple', edgecolor='white')
    axes[1,1].set_title('Répartition nb_salons')
    q_counts = train['quartier'].value_counts()
    axes[1,2].barh(q_counts.index, q_counts.values, color='coral', edgecolor='white')
    axes[1,2].set_title('Annonces par quartier')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'etape5_univariee.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 5 bivariée
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    order = train.groupby('quartier')['prix'].median().sort_values(ascending=False).index
    data_q = [train[train['quartier']==q]['prix'].values/1e6 for q in order]
    bp = axes[0,0].boxplot(data_q, tick_labels=order, patch_artist=True)
    [p.set(facecolor='steelblue', alpha=0.7) for p in bp['boxes']]
    axes[0,0].set_ylim(0, train['prix'].quantile(0.95)/1e6 * 1.1)
    axes[0,0].set_title('Prix par quartier'); axes[0,0].tick_params(axis='x', rotation=45)
    top4 = train['quartier'].value_counts().head(4).index
    pal = ['steelblue','darkorange','mediumseagreen','mediumpurple']
    for q, c in zip(top4, pal):
        g = train[train['quartier']==q]
        axes[0,1].scatter(g['surface_m2'], g['prix']/1e6, alpha=0.5, s=25, label=q, color=c)
    axes[0,1].set_title('Prix vs Surface (top 4 quartiers)')
    axes[0,1].legend(fontsize=9)
    train_ch = train[train['nb_chambres']<=10]
    order_ch = sorted(train_ch['nb_chambres'].unique())
    data_ch = [train_ch[train_ch['nb_chambres']==c]['prix'].values/1e6 for c in order_ch]
    bp2 = axes[1,0].boxplot(data_ch, tick_labels=[str(int(c)) for c in order_ch], patch_artist=True)
    [p.set(facecolor='darkorange', alpha=0.7) for p in bp2['boxes']]
    axes[1,0].set_title('Prix par nb_chambres')
    med_q = train.groupby('quartier')['prix'].median().sort_values(ascending=False)/1e6
    bars = axes[1,1].barh(med_q.index, med_q.values, color='teal', edgecolor='white')
    for bar, val in zip(bars, med_q.values):
        axes[1,1].text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                       f'{val:.1f}M', va='center', fontsize=9)
    axes[1,1].set_title('Prix médian par quartier')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'etape5_bivariee.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Heatmap corrélation
    corr_df = train[['prix','surface_m2','nb_chambres','nb_salons','nb_sdb']].dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Matrice de corrélation')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'etape5_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("Figures EDA sauvegardées.")


def step6_feature_engineering(train, test, imputer):
    print("\n" + "="*60)
    print("ÉTAPE 6.1-6.3 : Feature Engineering")
    print("="*60)

    # Géo-enrichissement
    train = enrich_geo(train)
    test  = enrich_geo(test)
    print("Géo-enrichissement terminé.")

    # Encodage quartier (sur le train uniquement)
    quartier_enc = QuartierEncoder()
    quartier_enc.fit(train, target_col='prix')

    # Features (passe 1 pour calculer médiane taille_rue)
    train_feat = build_features(train, imputer, quartier_enc, is_train=True)
    imputer.fit_taille_rue(train_feat)

    # Features (passe 2 avec taille_rue)
    train_feat = build_features(train, imputer, quartier_enc, is_train=True)
    imputer.fit_feature_medians(train_feat)
    train_feat = imputer.apply_feature_medians(train_feat)
    test_feat  = build_features(test,  imputer, quartier_enc, is_train=False)

    # Vérifications
    nan_train = train_feat[FEATURE_COLS].isna().sum().sum()
    nan_test  = test_feat[FEATURE_COLS].isna().sum().sum()
    print(f"NaN dans features train : {nan_train}")
    print(f"NaN dans features test  : {nan_test}")

    # Sauvegarder
    train_feat.to_csv(os.path.join(BASE_DIR, 'data', 'processed', 'enriched_data.csv'), index=False)
    test_feat.to_csv(os.path.join(BASE_DIR, 'data', 'processed', 'test_enriched.csv'),  index=False)
    print("Données enrichies sauvegardées.")

    return train_feat, test_feat


def step6_modeling(train_feat, test_feat):
    print("\n" + "="*60)
    print("ÉTAPE 6.4-6.6 : Modélisation")
    print("="*60)

    X_train = train_feat[FEATURE_COLS].values
    y_train = np.log1p(train_feat['prix'].values)
    X_test  = test_feat[FEATURE_COLS].values

    # Baseline
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    baseline_scores = []
    for tr_idx, val_idx in kf.split(X_train):
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        median_pred = np.full_like(y_val, np.median(y_tr))
        baseline_scores.append(rmsle_original_scale(y_val, median_pred))
    baseline_score = np.mean(baseline_scores)
    print(f"Baseline RMSLE : {baseline_score:.4f}")

    # Comparaison
    results_df = compare_all_models(X_train, y_train, baseline_score)
    print("\nRésultats :")
    print(results_df[['RMSLE', 'R2']].round(4))

    # Figure comparaison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    all_m = results_df.copy()
    colors_bar = ['red' if idx=='Baseline (médiane)' else 'steelblue' for idx in all_m.index]
    axes[0].bar(all_m.index, all_m['RMSLE'], color=colors_bar, edgecolor='white', alpha=0.85)
    axes[0].axhline(baseline_score, color='red', linestyle='--', lw=1.5)
    axes[0].set_title('RMSLE (↓ meilleur)'); axes[0].tick_params(axis='x', rotation=45)
    non_bl = results_df.drop('Baseline (médiane)', errors='ignore')
    axes[1].bar(non_bl.index, non_bl['R2'].fillna(0), color='teal', edgecolor='white', alpha=0.85)
    axes[1].set_title('R² (↑ meilleur)'); axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'etape6_comparaison_modeles.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Meilleur modèle
    non_bl = results_df.drop('Baseline (médiane)', errors='ignore')
    best_name = non_bl['RMSLE'].idxmin()
    best_cv_rmsle = non_bl.loc[best_name, 'RMSLE']
    print(f"\nMeilleur modèle : {best_name} (RMSLE CV = {best_cv_rmsle:.4f})")

    models_dict = get_models()
    best_model = train_final_model(models_dict[best_name], X_train, y_train)

    # Feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    if hasattr(best_model, 'feature_importances_'):
        fi = pd.DataFrame({'feature': FEATURE_COLS, 'importance': best_model.feature_importances_})
        fi = fi.sort_values('importance').tail(20)
        ax.barh(fi['feature'], fi['importance'], color='steelblue', edgecolor='white')
        ax.set_title(f'Top 20 Feature Importances — {best_name}')
    elif hasattr(best_model, 'coef_'):
        fi = pd.DataFrame({'feature': FEATURE_COLS, 'importance': np.abs(best_model.coef_)})
        fi = fi.sort_values('importance').tail(20)
        ax.barh(fi['feature'], fi['importance'], color='steelblue', edgecolor='white')
        ax.set_title(f'Top 20 Coefficients — {best_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'etape6_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Réel vs prédit
    y_pred_tr = best_model.predict(X_train)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_train, y_pred_tr, alpha=0.3, s=15, color='steelblue')
    mv = [min(y_train.min(), y_pred_tr.min()), max(y_train.max(), y_pred_tr.max())]
    ax.plot(mv, mv, 'r-', lw=2, label='y=x')
    ax.set_xlabel('log(1+prix) réel'); ax.set_ylabel('log(1+prix) prédit')
    ax.set_title(f'Réel vs Prédit — {best_name}')
    rmsle_tr = rmsle_original_scale(y_train, y_pred_tr)
    ax.text(0.05, 0.95, f'RMSLE: {rmsle_tr:.4f}', transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'etape6_reel_vs_predit.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Soumission
    submission = generate_submission(
        best_model, X_test,
        test_ids=test_feat['id'],
        output_path=os.path.join(BASE_DIR, 'data', 'final', 'submission.csv')
    )

    # Sauvegarde modèle
    save_model(best_model, FEATURE_COLS, model_dir=os.path.join(BASE_DIR, 'model'))

    # Résumé
    improvement = (baseline_score - best_cv_rmsle) / baseline_score * 100
    print("\n" + "="*60)
    print("RÉSUMÉ FINAL")
    print("="*60)
    print(f"Baseline RMSLE           : {baseline_score:.4f}")
    print(f"Meilleur modèle          : {best_name}")
    print(f"RMSLE CV                 : {best_cv_rmsle:.4f}")
    print(f"Amélioration vs baseline : {improvement:.1f}%")
    print(f"Soumission               : data/final/submission.csv ({len(submission)} lignes)")
    print(f"Modèle                   : model/housing_model.pkl")
    print("="*60)

    return best_model, results_df


def main():
    print("PIPELINE CAPSTONE : Prédiction Prix Immobiliers — Nouakchott")

    train, test = step1_2_load_clean()
    train, test, imputer = step3_missing(train, test)
    step4_5_eda(train)
    train_feat, test_feat = step6_feature_engineering(train, test, imputer)
    step6_modeling(train_feat, test_feat)

    print("\nPipeline terminé avec succès !")


if __name__ == '__main__':
    main()
