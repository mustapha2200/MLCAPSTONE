export interface Quartier {
  nom: string;
  latitude: number;
  longitude: number;
  prix_median: number;
  prix_moyen?: number;
  nb_annonces: number;
  surface_mediane?: number;
  caractere: string;
}

export interface PredictionRequest {
  surface_m2: number;
  nb_chambres: number;
  nb_salons: number;
  nb_sdb: number | null;
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
  distribution_prix: {
    bins: number[];
    counts: number[];
  };
  model_info: {
    nom: string;
    rmsle: number;
    r2: number;
    nb_features: number;
  };
}
