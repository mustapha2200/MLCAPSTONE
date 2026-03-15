'use client';

import { useState } from 'react';
import { PredictionRequest } from '@/types';
import { QUARTIERS } from '@/lib/constants';

interface Props {
  onSubmit: (data: PredictionRequest) => void;
  loading: boolean;
}

const defaultValues: PredictionRequest = {
  surface_m2:       150,
  nb_chambres:      3,
  nb_salons:        1,
  nb_sdb:           null,
  quartier:         'Teyarett',
  has_garage:       0,
  has_piscine:      0,
  has_clim:         0,
  has_titre_foncier: 0,
  description:      '',
};

export default function PredictionForm({ onSubmit, loading }: Props) {
  const [form, setForm] = useState<PredictionRequest>(defaultValues);

  const setField = (key: keyof PredictionRequest, val: number | string | null) =>
    setForm((prev) => ({ ...prev, [key]: val }));

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(form);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Quartier */}
      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-1">Quartier *</label>
        <select
          value={form.quartier}
          onChange={(e) => setField('quartier', e.target.value)}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-mauritania-green"
          required
        >
          {QUARTIERS.map((q) => (
            <option key={q.nom} value={q.nom}>
              {q.nom} — {q.caractere}
            </option>
          ))}
        </select>
      </div>

      {/* Surface + Chambres */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-1">Surface (m²) *</label>
          <input
            type="number"
            min={20} max={2000} step={5}
            value={form.surface_m2}
            onChange={(e) => setField('surface_m2', Number(e.target.value))}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-mauritania-green"
            required
          />
        </div>
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-1">Chambres *</label>
          <input
            type="number"
            min={0} max={15} step={1}
            value={form.nb_chambres}
            onChange={(e) => setField('nb_chambres', Number(e.target.value))}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-mauritania-green"
            required
          />
        </div>
      </div>

      {/* Salons + SDB */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-1">Salons *</label>
          <input
            type="number"
            min={0} max={10} step={1}
            value={form.nb_salons}
            onChange={(e) => setField('nb_salons', Number(e.target.value))}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-mauritania-green"
            required
          />
        </div>
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-1">
            Salles de bain <span className="text-gray-400 font-normal">(optionnel)</span>
          </label>
          <input
            type="number"
            min={0} max={10} step={1}
            value={form.nb_sdb ?? ''}
            placeholder="Non renseigné"
            onChange={(e) => setField('nb_sdb', e.target.value === '' ? null : Number(e.target.value))}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-mauritania-green"
          />
        </div>
      </div>

      {/* Checkboxes */}
      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-2">Équipements</label>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { key: 'has_garage',        label: '🚗 Garage' },
            { key: 'has_piscine',       label: '🏊 Piscine' },
            { key: 'has_clim',          label: '❄️ Climatisation' },
            { key: 'has_titre_foncier', label: '📜 Titre foncier' },
          ].map(({ key, label }) => (
            <label
              key={key}
              className="flex items-center gap-2 cursor-pointer bg-gray-50 hover:bg-gray-100 rounded-lg p-3 border border-gray-200 transition-colors"
            >
              <input
                type="checkbox"
                checked={form[key as keyof PredictionRequest] === 1}
                onChange={(e) => setField(key as keyof PredictionRequest, e.target.checked ? 1 : 0)}
                className="accent-mauritania-green w-4 h-4"
              />
              <span className="text-sm">{label}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Description */}
      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-1">
          Description <span className="text-gray-400 font-normal">(optionnel)</span>
        </label>
        <textarea
          rows={3}
          value={form.description}
          onChange={(e) => setField('description', e.target.value)}
          placeholder="Ex : Villa moderne avec jardin, piscine, standing…"
          className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-mauritania-green resize-none"
        />
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-mauritania-green hover:bg-green-800 disabled:opacity-60 text-white font-bold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Calcul en cours…
          </>
        ) : (
          '✨ Estimer le prix'
        )}
      </button>
    </form>
  );
}
