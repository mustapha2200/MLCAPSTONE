'use client';

import { QUARTIERS } from '@/lib/constants';

interface Props {
  selected: string;
  onChange: (nom: string) => void;
}

export default function QuartierSelector({ selected, onChange }: Props) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
      {QUARTIERS.map((q) => (
        <button
          key={q.nom}
          onClick={() => onChange(q.nom)}
          className={`text-left rounded-lg px-3 py-2 text-sm transition-all border ${
            selected === q.nom
              ? 'bg-mauritania-green text-white border-mauritania-green font-semibold'
              : 'bg-white text-gray-700 border-gray-200 hover:border-mauritania-green hover:text-mauritania-green'
          }`}
        >
          <div className="font-medium">{q.nom}</div>
          <div className="text-xs opacity-70 truncate">{q.caractere}</div>
        </button>
      ))}
    </div>
  );
}
