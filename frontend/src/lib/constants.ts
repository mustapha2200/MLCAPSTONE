export const QUARTIERS = [
  { nom: 'Tevragh Zeina', lat: 18.1036, lon: -15.9785, caractere: 'Quartier huppé, ambassades, villas' },
  { nom: 'Sebkha',        lat: 18.0730, lon: -15.9870, caractere: 'Commercial, marchés' },
  { nom: 'Ksar',          lat: 18.0866, lon: -15.9750, caractere: 'Centre historique' },
  { nom: 'Teyarett',      lat: 18.0950, lon: -15.9700, caractere: 'Centre, mixte' },
  { nom: 'Dar Naim',      lat: 18.1200, lon: -15.9450, caractere: 'Résidentiel, expansion' },
  { nom: 'Arafat',        lat: 18.0550, lon: -15.9610, caractere: 'Populaire, dense' },
  { nom: 'Toujounine',    lat: 18.0680, lon: -15.9350, caractere: 'Périphérie, récent' },
  { nom: 'Riyadh',        lat: 18.0850, lon: -15.9550, caractere: 'Résidentiel moyen' },
];

export const NOUAKCHOTT_CENTER = { lat: 18.0866, lon: -15.975 };
export const MRU_TO_EUR = 1 / 400;

export const formatMRU = (val: number): string => {
  if (val >= 1_000_000) return `${(val / 1_000_000).toFixed(1)}M MRU`;
  if (val >= 1_000)     return `${(val / 1_000).toFixed(0)}K MRU`;
  return `${val.toFixed(0)} MRU`;
};

export const formatEUR = (val: number): string =>
  new Intl.NumberFormat('fr-FR', { style: 'currency', currency: 'EUR', maximumFractionDigits: 0 }).format(val);
