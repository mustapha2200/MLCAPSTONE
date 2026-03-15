'use client';

import dynamic from 'next/dynamic';

const LeafletMap = dynamic(() => import('./LeafletMap'), {
  ssr: false,
  loading: () => (
    <div className="h-[400px] bg-gray-100 animate-pulse rounded-lg flex items-center justify-center text-gray-400">
      Chargement de la carte…
    </div>
  ),
});

interface QuartierMapItem {
  nom: string;
  lat: number;
  lon: number;
  prix_median?: number;
  caractere?: string;
}

interface Props {
  quartiers: QuartierMapItem[];
  selected?: string;
  center?: { lat: number; lon: number };
  zoom?: number;
}

export default function MapComponent({ quartiers, selected, center, zoom }: Props) {
  return <LeafletMap quartiers={quartiers} selected={selected} center={center} zoom={zoom} />;
}
