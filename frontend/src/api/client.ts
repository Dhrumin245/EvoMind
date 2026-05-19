import axios from 'axios';
import { useAuthStore } from '../store/authStore';

const envBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api';

export const apiClient = axios.create({
  baseURL: envBaseUrl,
  timeout: 20000,
});

apiClient.interceptors.request.use((config) => {
  const storedKey = useAuthStore.getState().apiKey;
  const sessionToken = useAuthStore.getState().sessionToken;
  const envKey = import.meta.env.VITE_API_KEY;
  const apiKey = storedKey || envKey;

  if (apiKey) {
    config.headers.set('X-API-Key', apiKey);
  } else if (sessionToken) {
    config.headers.set('Authorization', `Bearer ${sessionToken}`);
  }
  return config;
});

export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === 'string') {
      return detail;
    }
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'Unexpected error';
}
