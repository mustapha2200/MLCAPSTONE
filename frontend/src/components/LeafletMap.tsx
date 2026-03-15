'use client';

import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { formatMRU } from '@/lib/constants';

// Fix icônes Leaflet dans Next.js
// eslint-disable-next-line @typescript-eslint/no-explicit-any
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl:       'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl:     'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
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

export default function LeafletMap({ quartiers, selected, center, zoom = 12 }: Props) {
  const mapCenter = center ?? { lat: 18.0866, lon: -15.975 };

  const selectedIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-gold.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
  });

  return (
    <MapContainer
      center={[mapCenter.lat, mapCenter.lon]}
      zoom={zoom}
      className="h-[400px] w-full rounded-lg z-0"
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
      />

      {quartiers.map((q) => (
        <Marker
          key={q.nom}
          position={[q.lat, q.lon]}
          icon={q.nom === selected ? selectedIcon : new L.Icon.Default()}
        >
          <Popup>
            <div className="text-sm">
              <strong className="text-mauritania-green">{q.nom}</strong>
              {q.prix_median && (
                <div>Prix médian : <strong>{formatMRU(q.prix_median)}</strong></div>
              )}
              {q.caractere && <div className="text-gray-500">{q.caractere}</div>}
            </div>
          </Popup>
        </Marker>
      ))}

      {quartiers
        .filter((q) => q.prix_median)
        .map((q) => (
          <Circle
            key={`circle-${q.nom}`}
            center={[q.lat, q.lon]}
            radius={(q.prix_median ?? 1_000_000) / 3000}
            pathOptions={{
              color:       q.nom === selected ? '#FCD116' : '#006233',
              fillColor:   q.nom === selected ? '#FCD116' : '#006233',
              fillOpacity: 0.25,
              weight:      2,
            }}
          />
        ))}
    </MapContainer>
  );
}
