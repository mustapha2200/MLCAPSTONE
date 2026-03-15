'use client';

import { MarketStats } from '@/types';
import { formatMRU, MRU_TO_EUR, formatEUR } from '@/lib/constants';

interface Props {
  stats: MarketStats | null;
}

export default function StatsCards({ stats }: Props) {
  const cards = [
    {
      label: 'Annonces analysées',
      value: stats ? stats.total_annonces.toLocaleString('fr-FR') : '—',
      sub: 'données réelles du marché',
      icon: '📊',
      color: 'border-mauritania-green',
    },
    {
      label: 'Prix médian',
      value: stats ? formatMRU(stats.prix_median) : '—',
      sub: stats ? formatEUR(stats.prix_median * MRU_TO_EUR) : '',
      icon: '💰',
      color: 'border-mauritania-gold',
    },
    {
      label: 'Surface médiane',
      value: stats ? `${stats.surface_mediane} m²` : '—',
      sub: 'superficie typique',
      icon: '📐',
      color: 'border-mauritania-green',
    },
    {
      label: 'Précision du modèle',
      value: stats ? `RMSLE ${stats.model_info.rmsle}` : '—',
      sub: `R² = ${stats?.model_info.r2 ?? '—'} · ${stats?.model_info.nb_features ?? 34} features`,
      icon: '🤖',
      color: 'border-mauritania-gold',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card) => (
        <div
          key={card.label}
          className={`bg-white rounded-xl shadow-md p-6 border-l-4 ${card.color} flex flex-col gap-1`}
        >
          <div className="text-2xl">{card.icon}</div>
          <div className="text-2xl font-bold text-gray-800">{card.value}</div>
          <div className="text-sm font-semibold text-gray-600">{card.label}</div>
          <div className="text-xs text-gray-400">{card.sub}</div>
        </div>
      ))}
    </div>
  );
}
