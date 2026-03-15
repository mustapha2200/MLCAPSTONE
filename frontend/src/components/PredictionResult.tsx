'use client';

import { PredictionResponse, Quartier } from '@/types';
import { formatMRU, formatEUR, MRU_TO_EUR } from '@/lib/constants';
import MapComponent from './MapComponent';

interface Props {
  result: PredictionResponse;
  quartiers: Quartier[];
}

export default function PredictionResult({ result, quartiers }: Props) {
  const q = quartiers.find((q) => q.nom === result.quartier);
  const center = q ? { lat: q.latitude, lon: q.longitude } : undefined;

  const isAboveMedian = result.prix_estime > result.comparable.prix_median_quartier;
  const diffPct = result.comparable.prix_median_quartier > 0
    ? Math.abs(
        ((result.prix_estime - result.comparable.prix_median_quartier) /
          result.comparable.prix_median_quartier) *
          100
      ).toFixed(0)
    : null;

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Prix principal */}
      <div className="bg-gradient-to-br from-mauritania-green to-green-800 rounded-xl p-6 text-white text-center shadow-lg">
        <div className="text-sm font-medium opacity-80 mb-1">Prix estimé</div>
        <div className="text-4xl font-bold mb-1">{formatMRU(result.prix_estime)}</div>
        <div className="text-lg opacity-80">{formatEUR(result.prix_estime * MRU_TO_EUR)}</div>

        <div className="mt-4 flex justify-center gap-6 text-sm opacity-90">
          <div>
            <div className="font-semibold">{formatMRU(result.intervalle.bas)}</div>
            <div className="opacity-70">Fourchette basse</div>
          </div>
          <div className="border-l border-white/30" />
          <div>
            <div className="font-semibold">{formatMRU(result.intervalle.haut)}</div>
            <div className="opacity-70">Fourchette haute</div>
          </div>
        </div>
      </div>

      {/* Détails */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <div className="bg-white rounded-xl shadow-sm p-4 border border-gray-100">
          <div className="text-xs text-gray-500 mb-1">Prix au m²</div>
          <div className="text-lg font-bold text-gray-800">{result.prix_m2.toLocaleString('fr-FR')} MRU</div>
        </div>
        <div className="bg-white rounded-xl shadow-sm p-4 border border-gray-100">
          <div className="text-xs text-gray-500 mb-1">Médiane {result.quartier}</div>
          <div className="text-lg font-bold text-gray-800">
            {formatMRU(result.comparable.prix_median_quartier)}
          </div>
        </div>
        <div className="bg-white rounded-xl shadow-sm p-4 border border-gray-100">
          <div className="text-xs text-gray-500 mb-1">Annonces dans le quartier</div>
          <div className="text-lg font-bold text-gray-800">{result.comparable.nb_annonces_quartier}</div>
        </div>
      </div>

      {/* Badge comparaison */}
      {diffPct && (
        <div
          className={`rounded-lg px-4 py-3 text-sm font-medium flex items-center gap-2 ${
            isAboveMedian
              ? 'bg-amber-50 text-amber-800 border border-amber-200'
              : 'bg-green-50 text-mauritania-green border border-green-200'
          }`}
        >
          <span>{isAboveMedian ? '📈' : '📉'}</span>
          <span>
            {isAboveMedian
              ? `${diffPct}% au-dessus de la médiane de ${result.quartier}`
              : `${diffPct}% en-dessous de la médiane de ${result.quartier}`}
          </span>
        </div>
      )}

      {/* Carte */}
      {q && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Localisation — {result.quartier}</h3>
          <MapComponent
            quartiers={quartiers.map((qq) => ({
              nom: qq.nom,
              lat: qq.latitude,
              lon: qq.longitude,
              caractere: qq.caractere,
            }))}
            selected={result.quartier}
            center={center}
            zoom={14}
          />
        </div>
      )}
    </div>
  );
}
