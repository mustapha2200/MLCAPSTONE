# CLAUDE.md — Phase 5 : API Flask + Application Next.js

## Contexte

Ce fichier est la suite du projet Capstone "Prédiction des Prix Immobiliers en Mauritanie" (SupNum Master 1, 2026). Les phases 1-4 sont terminées. On a :

- Un modèle Random Forest entraîné : `model/housing_model.pkl` (RMSLE = 0.6576)
- La liste des features : `model/features.pkl`
- Les données enrichies : `data/processed/enriched_data.csv`
- Les coordonnées GPS des quartiers et les POI

**Cette phase vaut 20 points sur 100** (+ jusqu'à 7 points bonus).

## Ce qui existe déjà dans le repo

```
mauritania-housing-ml/
├── data/
│   ├── raw/kaggle_train.csv
│   ├── processed/enriched_data.csv
│   └── final/submission.csv
├── model/
│   ├── housing_model.pkl          ← modèle Random Forest sérialisé avec joblib
│   ├── features.pkl               ← liste ordonnée des 34+ feature names
│   └── scaler.pkl                 ← (si utilisé)
├── notebooks/
├── src/
│   ├── geo_enrichment.py
│   └── feature_engineering.py
└── outputs/figures/
```

## Ce que tu dois créer

```
mauritania-housing-ml/
├── api/
│   ├── app.py                     ← API Flask principale
│   ├── requirements.txt           ← dépendances Python backend
│   ├── config.py                  ← configuration (chemins, constantes)
│   ├── predict.py                 ← logique de prédiction isolée
│   └── Dockerfile                 ← (bonus) pour déploiement
├── frontend/
│   ├── package.json
│   ├── next.config.js
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── public/
│   │   └── favicon.ico
│   └── src/
│       ├── app/
│       │   ├── layout.tsx         ← layout principal
│       │   ├── page.tsx           ← page d'accueil
│       │   ├── predict/
│       │   │   └── page.tsx       ← formulaire de prédiction
│       │   ├── analysis/
│       │   │   └── page.tsx       ← (bonus) dashboard analytique
│       │   └── globals.css
│       ├── components/
│       │   ├── Navbar.tsx
│       │   ├── PredictionForm.tsx
│       │   ├── PredictionResult.tsx
│       │   ├── MapComponent.tsx   ← carte Leaflet
│       │   ├── StatsCards.tsx
│       │   └── QuartierSelector.tsx
│       ├── lib/
│       │   ├── api.ts             ← appels HTTP vers Flask
│       │   └── constants.ts       ← quartiers, GPS, etc.
│       └── types/
│           └── index.ts           ← types TypeScript
```

---

## PARTIE 1 : API Flask (Backend)

### 1.1 `api/app.py` — API principale

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # autoriser le frontend Next.js

# Charger le modèle et les features au démarrage
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
model = joblib.load(os.path.join(MODEL_DIR, 'housing_model.pkl'))
feature_cols = joblib.load(os.path.join(MODEL_DIR, 'features.pkl'))

# Charger les stats du train pour la page d'accueil
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
train = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'kaggle_train.csv'))
```

### 1.2 Endpoints à implémenter

#### `GET /api/health`
Simple health check.
```json
{"status": "ok", "model": "Random Forest", "features": 34}
```

#### `POST /api/predict`
Reçoit les caractéristiques d'un bien, retourne le prix estimé.

**Request body :**
```json
{
  "surface_m2": 250,
  "nb_chambres": 4,
  "nb_salons": 2,
  "nb_sdb": 2,
  "quartier": "Tevragh Zeina",
  "has_garage": 1,
  "has_piscine": 0,
  "has_clim": 1,
  "has_titre_foncier": 1,
  "description": "Villa moderne avec jardin"
}
```

**Logique du endpoint :**
1. Recevoir le JSON
2. Construire le vecteur de features dans le même ordre que `features.pkl`
3. Pour les features géo (latitude, longitude, distances, POI) → les dériver du quartier
4. Pour les features NLP → les calculer depuis la description
5. Pour les features manquantes → utiliser des valeurs par défaut raisonnables
6. Prédire avec le modèle : `log_prix = model.predict(X)[0]`
7. Reconvertir : `prix = np.expm1(log_prix)`
8. Calculer le prix/m² et comparer à la moyenne du quartier

**Response :**
```json
{
  "prix_estime": 8500000,
  "prix_estime_eur": 21250,
  "prix_m2": 34000,
  "prix_m2_quartier_moyen": 31000,
  "intervalle": {
    "bas": 6800000,
    "haut": 10200000
  },
  "quartier": "Tevragh Zeina",
  "comparable": {
    "prix_median_quartier": 6500000,
    "nb_annonces_quartier": 373,
    "surface_mediane_quartier": 300
  }
}
```

**Calcul de l'intervalle de confiance :**
- Si le modèle est un Random Forest, utiliser les prédictions individuelles de chaque arbre :
```python
if hasattr(model, 'estimators_'):
    tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
    tree_preds_prix = np.expm1(tree_preds)
    intervalle = {
        'bas': float(np.percentile(tree_preds_prix, 10)),
        'haut': float(np.percentile(tree_preds_prix, 90))
    }
```
- Sinon, utiliser ±20% du prix estimé comme approximation.

#### `GET /api/stats`
Retourne les statistiques clés du marché pour la page d'accueil.

```json
{
  "total_annonces": 1153,
  "prix_median": 2600000,
  "prix_moyen": 4339931,
  "surface_mediane": 200,
  "quartiers": [
    {
      "nom": "Tevragh Zeina",
      "prix_median": 6500000,
      "nb_annonces": 373,
      "surface_mediane": 300,
      "latitude": 18.1036,
      "longitude": -15.9785
    }
  ],
  "distribution_prix": {
    "bins": [0, 1, 2, 3, 5, 8, 10, 15, 20, 55],
    "counts": [45, 230, 180, 250, 200, 100, 80, 40, 28]
  },
  "model_info": {
    "nom": "Random Forest",
    "rmsle": 0.6576,
    "r2": 0.564,
    "nb_features": 34
  }
}
```

#### `GET /api/quartiers`
Retourne la liste des quartiers avec GPS et stats.

```json
[
  {
    "nom": "Tevragh Zeina",
    "latitude": 18.1036,
    "longitude": -15.9785,
    "prix_median": 6500000,
    "nb_annonces": 373,
    "caractere": "Quartier huppé, ambassades, villas"
  }
]
```

### 1.3 Mapping quartier → features géo

Le endpoint `/api/predict` doit reconstruire toutes les features géo à partir du nom du quartier :

```python
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

REFERENCE_POINTS = {
    'centre':   (18.0866, -15.9750),
    'aeroport': (18.0987, -15.9476),
    'plage':    (18.0580, -16.0200),
    'port':     (18.0340, -16.0280),
    'marche':   (18.0860, -15.9760),
}
```

Les POI par quartier doivent aussi être mappés (depuis `data/processed/poi_data.csv` ou hardcodés en fallback).

Le target encoding et frequency encoding doivent être calculés depuis le train au démarrage de l'API.

### 1.4 `api/requirements.txt`

```
flask>=3.0
flask-cors>=4.0
joblib>=1.3
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
gunicorn>=21.2
```

### 1.5 Lancement

```bash
cd api
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

Ou avec gunicorn :
```bash
gunicorn app:app --bind 0.0.0.0:5000
```

---

## PARTIE 2 : Frontend Next.js

### 2.1 Setup du projet

```bash
cd frontend
npx create-next-app@latest . --typescript --tailwind --eslint --app --src-dir
npm install axios leaflet react-leaflet @types/leaflet recharts
```

### 2.2 Configuration

**`next.config.js` :**
```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:5000/api/:path*',
      },
    ];
  },
};
module.exports = nextConfig;
```

### 2.3 Types TypeScript (`src/types/index.ts`)

```typescript
export interface Quartier {
  nom: string;
  latitude: number;
  longitude: number;
  prix_median: number;
  nb_annonces: number;
  caractere: string;
}

export interface PredictionRequest {
  surface_m2: number;
  nb_chambres: number;
  nb_salons: number;
  nb_sdb: number;
  quartier: string;
  has_garage: number;
  has_piscine: number;
  has_clim: number;
  has_titre_foncier: number;
  description: string;
}

export interface PredictionResponse {
  prix_estime: number;
  prix_estime_eur: number;
  prix_m2: number;
  prix_m2_quartier_moyen: number;
  intervalle: {
    bas: number;
    haut: number;
  };
  quartier: string;
  comparable: {
    prix_median_quartier: number;
    nb_annonces_quartier: number;
    surface_mediane_quartier: number;
  };
}

export interface MarketStats {
  total_annonces: number;
  prix_median: number;
  prix_moyen: number;
  surface_mediane: number;
  quartiers: Quartier[];
  model_info: {
    nom: string;
    rmsle: number;
    r2: number;
    nb_features: number;
  };
}
```

### 2.4 Constantes (`src/lib/constants.ts`)

```typescript
export const QUARTIERS = [
  { nom: "Tevragh Zeina", lat: 18.1036, lon: -15.9785, caractere: "Quartier huppé, ambassades" },
  { nom: "Ksar", lat: 18.0866, lon: -15.9750, caractere: "Centre historique" },
  { nom: "Teyarett", lat: 18.0950, lon: -15.9700, caractere: "Centre, mixte" },
  { nom: "Sebkha", lat: 18.0730, lon: -15.9870, caractere: "Commercial, marchés" },
  { nom: "Arafat", lat: 18.0550, lon: -15.9610, caractere: "Populaire, dense" },
  { nom: "Dar Naim", lat: 18.1200, lon: -15.9450, caractere: "Résidentiel, expansion" },
  { nom: "Toujounine", lat: 18.0680, lon: -15.9350, caractere: "Périphérie, récent" },
  { nom: "Riyadh", lat: 18.0850, lon: -15.9550, caractere: "Résidentiel moyen" },
];

export const NOUAKCHOTT_CENTER = { lat: 18.0866, lon: -15.975 };
export const MRU_TO_EUR = 1 / 400;
```

### 2.5 API client (`src/lib/api.ts`)

```typescript
import axios from 'axios';
import { PredictionRequest, PredictionResponse, MarketStats } from '@/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '/api';

export async function predict(data: PredictionRequest): Promise<PredictionResponse> {
  const res = await axios.post(`${API_BASE}/predict`, data);
  return res.data;
}

export async function getStats(): Promise<MarketStats> {
  const res = await axios.get(`${API_BASE}/stats`);
  return res.data;
}

export async function getQuartiers() {
  const res = await axios.get(`${API_BASE}/quartiers`);
  return res.data;
}
```

---

### 2.6 Pages à implémenter

#### Page d'accueil (`src/app/page.tsx`)

**Contenu :**
- Titre : "🏠 Prédiction des Prix Immobiliers — Nouakchott"
- Sous-titre : "Estimez la valeur de votre bien grâce au Machine Learning"
- **4 cartes statistiques** (StatsCards) :
  - Nombre total d'annonces analysées (1153)
  - Prix médian du marché (2.6M MRU / 6,500 EUR)
  - Surface médiane (200 m²)
  - Précision du modèle (R² = 0.564, RMSLE = 0.658)
- **Carte Leaflet** de Nouakchott avec les 8 quartiers (markers + cercles proportionnels au prix médian)
- **Bar chart** (recharts) des prix médians par quartier
- Bouton CTA : "Estimer un bien →" qui redirige vers `/predict`
- Section "À propos" : contexte du projet, SupNum, ML

**Style :** Tailwind, couleurs inspirées du drapeau mauritanien (vert #006233, or #FCD116), fond blanc, cartes avec ombre.

#### Page Prédiction (`src/app/predict/page.tsx`)

**Formulaire (PredictionForm) avec les champs :**

| Champ | Type | Valeurs |
|-------|------|---------|
| Quartier | dropdown | 8 quartiers |
| Surface (m²) | number input | min 20, max 2000, step 10 |
| Nombre de chambres | number input | min 0, max 15 |
| Nombre de salons | number input | min 0, max 10 |
| Nombre de salles de bain | number input | min 0, max 10 |
| Garage | checkbox | oui/non |
| Piscine | checkbox | oui/non |
| Climatisation | checkbox | oui/non |
| Titre foncier | checkbox | oui/non |
| Description (optionnel) | textarea | texte libre |

**Bouton "Estimer le prix"** → appel POST `/api/predict`

**Résultat (PredictionResult) affiché après la prédiction :**
- Prix estimé en gros : **8,500,000 MRU** (21,250 EUR)
- Intervalle de confiance : 6.8M — 10.2M MRU
- Prix au m² : 34,000 MRU/m²
- Comparaison : "Le prix médian à Tevragh Zeina est de 6,500,000 MRU"
- Badge : "Au-dessus de la médiane du quartier" ou "En-dessous"
- **Carte Leaflet** centrée sur le quartier sélectionné avec un marker

**UX :**
- Loading spinner pendant la prédiction
- Animation du résultat (fade-in)
- Possibilité de modifier et re-prédire sans recharger la page

#### Page Analyse — BONUS (`src/app/analysis/page.tsx`)

Si implémentée (+2 points) :
- Distribution des prix (histogramme recharts)
- Prix par quartier (bar chart)
- Surface vs Prix (scatter)
- Top features du modèle (bar chart horizontal)
- Comparaison par quartier (tableau interactif)

---

### 2.7 Composant carte Leaflet (`src/components/MapComponent.tsx`)

**IMPORTANT** : Leaflet en Next.js nécessite un import dynamique (pas de SSR).

Créer deux fichiers :

```typescript
// src/components/MapComponent.tsx
'use client';

import dynamic from 'next/dynamic';

const LeafletMap = dynamic(() => import('./LeafletMap'), {
  ssr: false,
  loading: () => <div className="h-[400px] bg-gray-100 animate-pulse rounded-lg" />,
});

interface Props {
  quartiers: Array<{ nom: string; lat: number; lon: number; prix_median?: number; caractere?: string }>;
  selected?: string;
  center?: { lat: number; lon: number };
  zoom?: number;
}

export default function MapComponent({ quartiers, selected, center, zoom }: Props) {
  return <LeafletMap quartiers={quartiers} selected={selected} center={center} zoom={zoom} />;
}
```

```typescript
// src/components/LeafletMap.tsx  (chargé dynamiquement, jamais SSR)
import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix icônes Leaflet en Next.js
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

interface Props {
  quartiers: Array<{ nom: string; lat: number; lon: number; prix_median?: number; caractere?: string }>;
  selected?: string;
  center?: { lat: number; lon: number };
  zoom?: number;
}

export default function LeafletMap({ quartiers, selected, center, zoom = 13 }: Props) {
  const mapCenter = center || { lat: 18.0866, lon: -15.975 };

  return (
    <MapContainer
      center={[mapCenter.lat, mapCenter.lon]}
      zoom={zoom}
      className="h-[400px] w-full rounded-lg z-0"
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; OpenStreetMap contributors'
      />
      {quartiers.map((q) => (
        <Marker key={q.nom} position={[q.lat, q.lon]}>
          <Popup>
            <strong>{q.nom}</strong><br />
            {q.prix_median && <>Prix médian: {(q.prix_median / 1e6).toFixed(1)}M MRU<br /></>}
            {q.caractere}
          </Popup>
        </Marker>
      ))}
      {quartiers.filter(q => q.prix_median).map((q) => (
        <Circle
          key={`circle-${q.nom}`}
          center={[q.lat, q.lon]}
          radius={(q.prix_median || 1000000) / 5000}
          pathOptions={{
            color: q.nom === selected ? '#FCD116' : '#006233',
            fillOpacity: 0.3,
          }}
        />
      ))}
    </MapContainer>
  );
}
```

**CSS** — ajouter dans `src/app/globals.css` :
```css
@import 'leaflet/dist/leaflet.css';

.leaflet-container {
  z-index: 0;
}
```

---

### 2.8 Design et couleurs

**Palette drapeau mauritanien :**

```javascript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        mauritania: {
          green: '#006233',
          gold: '#FCD116',
          dark: '#1a1a2e',
        },
      },
    },
  },
};
```

**Principes :**
- Fond : blanc / gris très clair (#f8f9fa)
- Titres et accents principaux : vert #006233
- Accents secondaires / CTA / badges : or #FCD116
- Texte : gris foncé
- Cartes (components) avec `shadow-lg rounded-xl p-6`
- Typographie : Inter ou font système
- Responsive : mobile-first

---

## PARTIE 3 : Lancement et tests

### Démarrer le backend

```bash
cd api
pip install -r requirements.txt
python app.py
# → http://localhost:5000
# Tester :
curl http://localhost:5000/api/health
```

### Démarrer le frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### Tester la prédiction end-to-end

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "surface_m2": 250,
    "nb_chambres": 4,
    "nb_salons": 2,
    "nb_sdb": 2,
    "quartier": "Tevragh Zeina",
    "has_garage": 1,
    "has_piscine": 0,
    "has_clim": 1,
    "has_titre_foncier": 1,
    "description": "Villa moderne avec jardin"
  }'
```

Réponse attendue : un JSON avec `prix_estime` autour de 5-12M MRU pour Tevragh Zeina 250m² 4ch.

---

## PARTIE 4 : Déploiement (Bonus +3 points)

### Backend → Render

1. Créer un Web Service sur render.com
2. Root directory : `api/`
3. Build : `pip install -r requirements.txt`
4. Start : `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Ajouter les fichiers `model/` et `data/` nécessaires

**`api/Dockerfile` (alternative) :**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY api/ ./api/
COPY model/ ./model/
COPY data/raw/kaggle_train.csv ./data/raw/kaggle_train.csv
COPY data/processed/ ./data/processed/
RUN pip install -r api/requirements.txt
EXPOSE 5000
CMD ["gunicorn", "api.app:app", "--bind", "0.0.0.0:5000"]
```

### Frontend → Vercel

```bash
cd frontend
npx vercel
```

Variable d'environnement à configurer :
```
NEXT_PUBLIC_API_URL=https://ton-api.onrender.com/api
```

---

## Barème Phase 5 (rappel)

| Critère | Points |
|---------|--------|
| API fonctionnelle (requête → prédiction) | 6 |
| Interface utilisateur propre et intuitive | 6 |
| Carte interactive (Leaflet/OSM) | 4 |
| Affichage clair des résultats | 4 |
| **Total Phase 5** | **20** |

| Bonus | Points |
|-------|--------|
| Déploiement en ligne (Vercel + Render) | +3 |
| Heatmap des prix sur la carte | +2 |
| Dashboard analytique interactif | +2 |

---

## Checklist finale

- [ ] `api/app.py` démarre et répond sur `/api/health`
- [ ] `POST /api/predict` retourne un prix cohérent (pas NaN, pas négatif)
- [ ] `GET /api/stats` retourne les statistiques du marché
- [ ] `GET /api/quartiers` retourne la liste avec GPS
- [ ] `frontend/` démarre avec `npm run dev` sans erreur
- [ ] Page d'accueil : stats + carte Leaflet + bar chart + bouton CTA
- [ ] Page prédiction : formulaire complet → résultat avec prix + intervalle + carte
- [ ] La carte Leaflet fonctionne (tiles OSM, markers, cercles)
- [ ] Le design utilise les couleurs mauritaniennes (vert/or)
- [ ] Responsive (mobile ok)
- [ ] CORS configuré (Flask → Next.js)
- [ ] Pas de clé API exposée côté client
- [ ] (Bonus) Déployé sur Vercel + Render
- [ ] (Bonus) Heatmap des prix sur la carte
- [ ] (Bonus) Page `/analysis` avec graphiques recharts
