'use client';

import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Legend, Cell,
} from 'recharts';
import { getStats, getModelInfo } from '@/lib/api';
import { MarketStats, ModelInfo } from '@/types';
import { formatMRU } from '@/lib/constants';
import MapComponent from '@/components/MapComponent';

export default function AnalysisPage() {
  const [stats, setStats]         = useState<MarketStats | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    getStats().then(setStats).catch(() => {});
    getModelInfo().then(setModelInfo).catch(() => {});
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
        {modelInfo ? (
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={modelInfo.model_comparison}>
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 1.1]} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(v) => [typeof v === 'number' ? v.toFixed(3) : v, 'RMSLE']} />
              <Bar dataKey="rmsle" radius={[4, 4, 0, 0]} fill="#006233">
                {modelInfo.model_comparison.map((m, i) => (
                  <Cell
                    key={i}
                    fill={m.name === 'Baseline' ? '#ef4444' : m.name === 'Stacking' ? '#FCD116' : '#006233'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-[260px] flex items-center justify-center text-gray-400">Chargement…</div>
        )}
      </section>

      {/* Feature importance */}
      <section className="bg-white rounded-xl shadow-md p-5">
        <h2 className="font-semibold text-gray-800 mb-4">Top 20 features ({modelInfo?.model ?? '…'})</h2>
        {modelInfo ? (
          <ResponsiveContainer width="100%" height={420}>
            <BarChart data={[...modelInfo.feature_importances].reverse()} layout="vertical">
              <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => `${(v * 100).toFixed(1)}%`} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={150} />
              <Tooltip formatter={(v) => [typeof v === 'number' ? `${(v * 100).toFixed(2)}%` : v, 'Importance']} />
              <Bar dataKey="importance" fill="#006233" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-[420px] flex items-center justify-center text-gray-400">Chargement…</div>
        )}
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
            { label: 'Modèle',    value: modelInfo?.model ?? '…' },
            { label: 'RMSLE CV', value: modelInfo?.rmsle != null ? modelInfo.rmsle.toFixed(4) : '…' },
            { label: 'R²',        value: modelInfo?.r2 != null ? modelInfo.r2.toFixed(3) : '—' },
            { label: 'Features', value: modelInfo?.nb_features != null ? String(modelInfo.nb_features) : '…' },
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
