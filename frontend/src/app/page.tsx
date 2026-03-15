'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { getStats } from '@/lib/api';
import { MarketStats } from '@/types';
import { formatMRU, QUARTIERS } from '@/lib/constants';
import StatsCards from '@/components/StatsCards';
import MapComponent from '@/components/MapComponent';

export default function HomePage() {
  const [stats, setStats] = useState<MarketStats | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    getStats().then(setStats).catch(() => setError(true));
  }, []);

  const quartierChartData = (stats?.quartiers ?? [])
    .sort((a, b) => b.prix_median - a.prix_median)
    .map((q) => ({
      name: q.nom.replace('Tevragh Zeina', 'T. Zeina'),
      prix: Math.round((q.prix_median / 1_000_000) * 10) / 10,
    }));

  const mapQuartiers = QUARTIERS.map((q) => {
    const s = stats?.quartiers.find((sq) => sq.nom === q.nom);
    return { ...q, prix_median: s?.prix_median };
  });

  return (
    <div className="space-y-10">
      {/* Hero */}
      <section className="text-center py-10">
        <h1 className="text-4xl font-bold mb-3" style={{ color: '#006233' }}>
          🏠 Prédiction des Prix Immobiliers
        </h1>
        <h2 className="text-xl text-gray-600 mb-2">Nouakchott, Mauritanie</h2>
        <p className="text-gray-500 max-w-xl mx-auto mb-8">
          Estimez la valeur de votre bien grâce au Machine Learning —
          Random Forest entraîné sur 1 153 annonces réelles.
        </p>
        <Link
          href="/predict"
          className="inline-flex items-center gap-2 text-white font-bold py-3 px-8 rounded-full transition-colors shadow-md text-lg"
          style={{ backgroundColor: '#006233' }}
        >
          Estimer un bien →
        </Link>
      </section>

      {/* Alertes API */}
      {error && (
        <div className="bg-amber-50 border border-amber-200 text-amber-800 rounded-lg p-4 text-sm">
          ⚠️ L&apos;API Flask n&apos;est pas disponible. Lancez{' '}
          <code className="bg-amber-100 px-1 rounded">python api/app.py</code> puis rechargez.
        </div>
      )}

      {/* Stat cards */}
      <StatsCards stats={stats} />

      {/* Carte + Bar chart */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-md p-5">
          <h3 className="font-semibold mb-3" style={{ color: '#006233' }}>
            Carte des quartiers
          </h3>
          <MapComponent quartiers={mapQuartiers} />
        </div>

        <div className="bg-white rounded-xl shadow-md p-5">
          <h3 className="font-semibold mb-3" style={{ color: '#006233' }}>
            Prix médian par quartier (M MRU)
          </h3>
          {quartierChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={quartierChartData} layout="vertical">
                <XAxis type="number" tick={{ fontSize: 12 }} tickFormatter={(v) => `${v}M`} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={85} />
                <Tooltip formatter={(v) => [`${v}M MRU`, 'Prix médian']} />
                <Bar dataKey="prix" radius={[0, 4, 4, 0]}>
                  {quartierChartData.map((_, i) => (
                    <Cell key={i} fill={i === 0 ? '#FCD116' : '#006233'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[340px] flex items-center justify-center text-gray-400">
              {error ? 'API non disponible' : 'Chargement…'}
            </div>
          )}
        </div>
      </section>

      {/* Distribution des prix */}
      {stats?.distribution_prix && (
        <section className="bg-white rounded-xl shadow-md p-5">
          <h3 className="font-semibold mb-3" style={{ color: '#006233' }}>
            Distribution des prix (M MRU)
          </h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart
              data={stats.distribution_prix.counts.map((count, i) => ({
                bin: `${stats.distribution_prix.bins[i]}–${stats.distribution_prix.bins[i + 1]}`,
                count,
              }))}
            >
              <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => [v, 'Annonces']} />
              <Bar dataKey="count" fill="#006233" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </section>
      )}

      {/* À propos */}
      <section className="bg-white rounded-xl shadow-md p-6">
        <h3 className="font-semibold text-lg mb-3" style={{ color: '#006233' }}>
          À propos du projet
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm text-gray-600">
          <div>
            <div className="font-semibold text-gray-800 mb-1">🎓 Contexte académique</div>
            Projet Capstone Machine Learning — Master 1 SupNum, Mauritanie (2026).
            Compétition Kaggle interne évaluée sur le RMSLE.
          </div>
          <div>
            <div className="font-semibold text-gray-800 mb-1">📊 Données</div>
            1 153 annonces immobilières réelles scrappées sur voursa.com.
            8 quartiers de Nouakchott, prix entre 200K et 54M MRU.
          </div>
          <div>
            <div className="font-semibold text-gray-800 mb-1">🤖 Modèle</div>
            Random Forest avec 34 features (géo, texte, surface).
            RMSLE = 0.6576 vs baseline 0.997 (−34%).
          </div>
        </div>
      </section>

      {/* Tableau quartiers */}
      {stats && (
        <section className="bg-white rounded-xl shadow-md p-5 overflow-x-auto">
          <h3 className="font-semibold mb-3" style={{ color: '#006233' }}>
            Statistiques par quartier
          </h3>
          <table className="w-full text-sm text-left">
            <thead>
              <tr className="border-b border-gray-200 text-gray-500">
                <th className="py-2 pr-4 font-semibold">Quartier</th>
                <th className="py-2 pr-4 font-semibold text-right">Annonces</th>
                <th className="py-2 pr-4 font-semibold text-right">Prix médian</th>
                <th className="py-2 pr-4 font-semibold text-right">Surface médiane</th>
                <th className="py-2 font-semibold">Caractère</th>
              </tr>
            </thead>
            <tbody>
              {stats.quartiers
                .sort((a, b) => b.prix_median - a.prix_median)
                .map((q) => (
                  <tr key={q.nom} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-2 pr-4 font-medium text-gray-800">{q.nom}</td>
                    <td className="py-2 pr-4 text-right text-gray-600">{q.nb_annonces}</td>
                    <td className="py-2 pr-4 text-right font-semibold" style={{ color: '#006233' }}>
                      {formatMRU(q.prix_median)}
                    </td>
                    <td className="py-2 pr-4 text-right text-gray-600">{q.surface_mediane} m²</td>
                    <td className="py-2 text-gray-500 text-xs">{q.caractere}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
