'use client';

import { useState, useEffect } from 'react';
import { predict, getQuartiers } from '@/lib/api';
import { PredictionRequest, PredictionResponse, Quartier } from '@/types';
import PredictionForm from '@/components/PredictionForm';
import PredictionResult from '@/components/PredictionResult';

export default function PredictPage() {
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState<PredictionResponse | null>(null);
  const [error, setError]       = useState<string | null>(null);
  const [quartiers, setQuartiers] = useState<Quartier[]>([]);

  useEffect(() => {
    getQuartiers().then(setQuartiers).catch(() => {});
  }, []);

  const handleSubmit = async (data: PredictionRequest) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predict(data);
      setResult(res);
    } catch (e) {
      setError(
        "Impossible de contacter l'API. Assurez-vous que le backend Flask est démarré sur le port 5000."
      );
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-1" style={{ color: '#006233' }}>
          Estimer un bien immobilier
        </h1>
        <p className="text-gray-500">Renseignez les caractéristiques du bien pour obtenir une estimation de prix.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        {/* Formulaire */}
        <div className="bg-white rounded-xl shadow-md p-6">
          <h2 className="font-semibold text-gray-800 mb-4">Caractéristiques du bien</h2>
          <PredictionForm onSubmit={handleSubmit} loading={loading} quartiers={quartiers} />
        </div>

        {/* Résultat */}
        <div>
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 text-sm">
              ❌ {error}
            </div>
          )}

          {!result && !error && !loading && (
            <div className="bg-white rounded-xl shadow-md p-10 text-center text-gray-400 border-2 border-dashed border-gray-200">
              <div className="text-5xl mb-3">🏠</div>
              <div className="font-medium">Remplissez le formulaire</div>
              <div className="text-sm mt-1">Le prix estimé apparaîtra ici</div>
            </div>
          )}

          {loading && (
            <div className="bg-white rounded-xl shadow-md p-10 text-center text-gray-500">
              <div className="inline-block w-10 h-10 border-4 border-[#006233] border-t-transparent rounded-full animate-spin mb-4" />
              <div className="font-medium">Calcul en cours…</div>
              <div className="text-sm text-gray-400 mt-1">Le modèle analyse 34 features</div>
            </div>
          )}

          {result && <PredictionResult result={result} quartiers={quartiers} />}
        </div>
      </div>

      {/* Infos méthodologie */}
      <section className="bg-white rounded-xl shadow-md p-5 text-sm text-gray-600">
        <h3 className="font-semibold text-gray-800 mb-2">Comment fonctionne l&apos;estimation ?</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div>
            <span className="font-medium text-[#006233]">📍 Géolocalisation</span><br />
            Les coordonnées GPS du quartier, les distances au centre-ville, à l&apos;aéroport et
            à la plage sont intégrées au modèle.
          </div>
          <div>
            <span className="font-medium text-[#006233]">📊 34 features</span><br />
            Surface, pièces, équipements, POI environnants (écoles, mosquées, commerces)
            et caractéristiques du texte de l&apos;annonce.
          </div>
          <div>
            <span className="font-medium text-[#006233]">🎯 Intervalle de confiance</span><br />
            Calculé à partir des 200 arbres du Random Forest (percentiles 10 et 90 des prédictions individuelles).
          </div>
        </div>
      </section>
    </div>
  );
}
