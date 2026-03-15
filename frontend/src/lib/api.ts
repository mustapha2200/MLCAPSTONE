import axios from 'axios';
import { PredictionRequest, PredictionResponse, MarketStats, Quartier } from '@/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://mlcapstone.onrender.com/api';

export async function predict(data: PredictionRequest): Promise<PredictionResponse> {
  const res = await axios.post(`${API_BASE}/predict`, data);
  return res.data;
}

export async function getStats(): Promise<MarketStats> {
  const res = await axios.get(`${API_BASE}/stats`);
  return res.data;
}

export async function getQuartiers(): Promise<Quartier[]> {
  const res = await axios.get(`${API_BASE}/quartiers`);
  return res.data;
}

export async function getModelInfo(): Promise<import('@/types').ModelInfo> {
  const res = await axios.get(`${API_BASE}/model-info?top=20`);
  return res.data;
}

export async function healthCheck(): Promise<boolean> {
  try {
    await axios.get(`${API_BASE}/health`);
    return true;
  } catch {
    return false;
  }
}
