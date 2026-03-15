// Constantes utilitaires — aucune donnée géographique ou statistique hardcodée.
// Les données des quartiers, prix, et features sont chargées depuis l'API Flask.

export const MRU_TO_EUR = 1 / 400;

export const NOUAKCHOTT_CENTER = { lat: 18.0866, lon: -15.975 };

export const formatMRU = (val: number): string => {
  if (val >= 1_000_000) return `${(val / 1_000_000).toFixed(1)}M MRU`;
  if (val >= 1_000)     return `${(val / 1_000).toFixed(0)}K MRU`;
  return `${val.toFixed(0)} MRU`;
};

export const formatEUR = (val: number): string =>
  new Intl.NumberFormat('fr-FR', { style: 'currency', currency: 'EUR', maximumFractionDigits: 0 }).format(val);
