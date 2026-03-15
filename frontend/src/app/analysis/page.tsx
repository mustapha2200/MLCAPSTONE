'use client';

import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Legend, Cell,
} from 'recharts';
import { getStats } from '@/lib/api';
import { MarketStats } from '@/types';
import { formatMRU } from '@/lib/constants';
import MapComponent from '@/components/MapComponent';

// Top 20 features du modèle Random Forest (valeurs approx. depuis feature_importance)
const TOP_FEATURES = [
  { name: 'quartier_target_enc', importance: 0.312 },
  { name: 'surface_m2',          importance: 0.198 },
  { name: 'log_surface',         importance: 0.087 },
  { name: 'nb_pieces_total',     importance: 0.062 },
  { name: 'nb_chambres',         importance: 0.047 },
  { name: 'dist_plage_km',       importance: 0.038 },
  { name: 'dist_centre_km',      importance: 0.033 },
  { name: 'latitude',            importance: 0.028 },
  { name: 'surface_par_piece',   importance: 0.025 },
  { name: 'nb_commerces_1km',    importance: 0.022 },
  { name: 'taille_rue',          importance: 0.019 },
  { name: 'quartier_freq',       importance: 0.016 },
  { name: 'nb_salons',           importance: 0.015 },
  { name: 'longitude',           importance: 0.013 },
  { name: 'age_annonce_jours',   importance: 0.011 },
  { name: 'nb_total_pois_1km',   importance: 0.010 },
  { name: 'desc_len',            importance: 0.009 },
  { name: 'has_titre_foncier',   importance: 0.008 },
  { name: 'dist_aeroport_km',    importance: 0.007 },
  { name: 'nb_sdb',              importance: 0.007 },
];

const MODEL_COMPARISON = [
  { name: 'Baseline',          rmsle: 0.999, r2: 0.00  },
  { name: 'Lin. Régression',   rmsle: 0.849, r2: 0.28  },
  { name: 'Ridge',             rmsle: 0.848, r2: 0.29  },
  { name: 'Lasso',             rmsle: 0.851, r2: 0.27  },
  { name: 'Random Forest',     rmsle: 0.658, r2: 0.564 },
  { name: 'Gradient Boosting', rmsle: 0.671, r2: 0.541 },
];

export default function AnalysisPage() {
  const [stats, setStats] = useState<MarketStats | null>(null);

  useEffect(() => {
    getStats().then(setStats).catch(() => {});
  }, []);

  const mapQuartiers = (stats?.quartiers ?? []).map((q) => ({
    nom: q.nom,
    lat: q.latitude,
    lon: q.longitude,
    prix_median: q.prix_median,
    caractere: q.caractere,
  }));

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-1" style={{ color: '#006233' }}>
          Analyse du marché immobilier
        </h1>
        <p className="text-gray-500">Exploration des données et performance du modèle.</p>
      </div>

      {/* Comparaison des modèles */}
      <section className="bg-white rounded-xl shadow-md p-5">
        <h2 className="font-semibold text-gray-800 mb-4">Comparaison des modèles — RMSLE (↓ meilleur)</h2>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={MODEL_COMPARISON}>
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 1.1]} tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v) => [typeof v === 'number' ? v.toFixed(3) : v, 'RMSLE']} />
            <Bar dataKey="rmsle" radius={[4, 4, 0, 0]}
              fill="#006233"
              label={{ position: 'right', fontSize: 11 }}
            >
              {MODEL_COMPARISON.map((m, i) => (
                <Cell
                  key={i}
                  fill={m.name === 'Baseline' ? '#ef4444' : m.name === 'Random Forest' ? '#FCD116' : '#006233'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </section>

      {/* Feature importance */}
      <section className="bg-white rounded-xl shadow-md p-5">
        <h2 className="font-semibold text-gray-800 mb-4">Top 20 features (Random Forest)</h2>
        <ResponsiveContainer width="100%" height={420}>
          <BarChart data={[...TOP_FEATURES].reverse()} layout="vertical">
            <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => `${(v * 100).toFixed(1)}%`} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={150} />
            <Tooltip formatter={(v) => [typeof v === 'number' ? `${(v * 100).toFixed(2)}%` : v, 'Importance']} />
            <Bar dataKey="importance" fill="#006233" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </section>

      {/* Prix par quartier + carte */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-md p-5">
          <h2 className="font-semibold text-gray-800 mb-4">Prix médian vs moyen par quartier</h2>
          {stats ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={stats.quartiers
                  .sort((a, b) => b.prix_median - a.prix_median)
                  .map((q) => ({
                    name:   q.nom.replace('Tevragh Zeina', 'T. Zeina'),
                    median: Math.round(q.prix_median / 1e6 * 10) / 10,
                    moyen:  Math.round((q.prix_moyen ?? 0) / 1e6 * 10) / 10,
                  }))}
                layout="vertical"
              >
                <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => `${v}M`} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={80} />
                <Tooltip formatter={(v) => [`${v}M MRU`]} />
                <Legend />
                <Bar dataKey="median" name="Médian" fill="#006233" radius={[0, 2, 2, 0]} />
                <Bar dataKey="moyen"  name="Moyen"  fill="#FCD116" radius={[0, 2, 2, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[300px] flex items-center justify-center text-gray-400">Chargement…</div>
          )}
        </div>

        <div className="bg-white rounded-xl shadow-md p-5">
          <h2 className="font-semibold text-gray-800 mb-4">Carte des prix par quartier</h2>
          <MapComponent quartiers={mapQuartiers} />
        </div>
      </section>

      {/* Tableau détaillé */}
      {stats && (
        <section className="bg-white rounded-xl shadow-md p-5 overflow-x-auto">
          <h2 className="font-semibold text-gray-800 mb-4">Statistiques détaillées par quartier</h2>
          <table className="w-full text-sm text-left">
            <thead>
              <tr className="border-b border-gray-200 text-gray-500 text-xs uppercase">
                <th className="py-2 pr-4">Quartier</th>
                <th className="py-2 pr-4 text-right">Annonces</th>
                <th className="py-2 pr-4 text-right">Prix médian</th>
                <th className="py-2 pr-4 text-right">Prix moyen</th>
                <th className="py-2 pr-4 text-right">Surface médiane</th>
                <th className="py-2">Caractère</th>
              </tr>
            </thead>
            <tbody>
              {stats.quartiers
                .sort((a, b) => b.prix_median - a.prix_median)
                .map((q) => (
                  <tr key={q.nom} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-2 pr-4 font-semibold" style={{ color: '#006233' }}>{q.nom}</td>
                    <td className="py-2 pr-4 text-right text-gray-600">{q.nb_annonces}</td>
                    <td className="py-2 pr-4 text-right font-medium">{formatMRU(q.prix_median)}</td>
                    <td className="py-2 pr-4 text-right text-gray-500">{formatMRU(q.prix_moyen ?? 0)}</td>
                    <td className="py-2 pr-4 text-right text-gray-500">{q.surface_mediane} m²</td>
                    <td className="py-2 text-gray-400 text-xs">{q.caractere}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </section>
      )}

      {/* Info modèle */}
      <section className="bg-white rounded-xl shadow-md p-5">
        <h2 className="font-semibold text-gray-800 mb-3">Informations sur le modèle</h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center text-sm">
          {[
            { label: 'Modèle',    value: 'Random Forest' },
            { label: 'RMSLE CV', value: '0.6576' },
            { label: 'R²',        value: '0.564' },
            { label: 'Features', value: '34' },
          ].map(({ label, value }) => (
            <div key={label} className="bg-gray-50 rounded-lg p-3">
              <div className="text-2xl font-bold" style={{ color: '#006233' }}>{value}</div>
              <div className="text-gray-500 mt-1">{label}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
