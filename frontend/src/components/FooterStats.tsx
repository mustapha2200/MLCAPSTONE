'use client';

import { useEffect, useState } from 'react';
import { getModelInfo } from '@/lib/api';

export default function FooterStats() {
  const [rmsle, setRmsle] = useState<number | null>(null);

  useEffect(() => {
    getModelInfo().then((info) => setRmsle(info.rmsle)).catch(() => {});
  }, []);

  if (rmsle == null) return null;

  return (
    <span className="text-mauritania-green font-medium">RMSLE = {rmsle.toFixed(4)}</span>
  );
}
