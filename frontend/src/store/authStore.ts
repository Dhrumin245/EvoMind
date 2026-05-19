import { create } from 'zustand';

type ModalName = 'createJob' | 'checkpoint' | 'compareGenomes' | null;

interface ToastMessage {
  id: string;
  title: string;
  description?: string;
  tone?: 'default' | 'success' | 'error';
}

interface AuthState {
  apiKey: string;
  sessionToken: string;
  userEmail: string;
  tenantId: string;
  selectedJobId: string;
  autoRefreshMs: number;
  modal: ModalName;
  toasts: ToastMessage[];
  setApiKey: (apiKey: string) => void;
  setSession: (token: string, userEmail: string, tenantId: string) => void;
  clearAuth: () => void;
  setSelectedJobId: (jobId: string) => void;
  setAutoRefreshMs: (value: number) => void;
  openModal: (modal: ModalName) => void;
  closeModal: () => void;
  pushToast: (toast: Omit<ToastMessage, 'id'>) => void;
  dismissToast: (id: string) => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  apiKey: localStorage.getItem('evomind.apiKey') || '',
  sessionToken: localStorage.getItem('evomind.sessionToken') || '',
  userEmail: localStorage.getItem('evomind.userEmail') || '',
  tenantId: localStorage.getItem('evomind.tenantId') || '',
  selectedJobId: localStorage.getItem('evomind.selectedJobId') || '',
  autoRefreshMs: 7500,
  modal: null,
  toasts: [],
  setApiKey: (apiKey) => {
    const trimmed = apiKey.trim();
    if (trimmed) {
      localStorage.setItem('evomind.apiKey', trimmed);
    } else {
      localStorage.removeItem('evomind.apiKey');
    }
    set({ apiKey: trimmed });
  },
  setSession: (token, userEmail, tenantId) => {
    localStorage.setItem('evomind.sessionToken', token);
    localStorage.setItem('evomind.userEmail', userEmail);
    localStorage.setItem('evomind.tenantId', tenantId);
    localStorage.removeItem('evomind.apiKey');
    set({ apiKey: '', sessionToken: token, userEmail, tenantId });
  },
  clearAuth: () => {
    localStorage.removeItem('evomind.apiKey');
    localStorage.removeItem('evomind.sessionToken');
    localStorage.removeItem('evomind.userEmail');
    localStorage.removeItem('evomind.tenantId');
    localStorage.removeItem('evomind.selectedJobId');
    set({ apiKey: '', sessionToken: '', userEmail: '', tenantId: '', selectedJobId: '' });
  },
  setSelectedJobId: (jobId) => {
    const trimmed = jobId.trim();
    if (trimmed) {
      localStorage.setItem('evomind.selectedJobId', trimmed);
    } else {
      localStorage.removeItem('evomind.selectedJobId');
    }
    set({ selectedJobId: trimmed });
  },
  setAutoRefreshMs: (value) => set({ autoRefreshMs: value }),
  openModal: (modal) => set({ modal }),
  closeModal: () => set({ modal: null }),
  pushToast: (toast) =>
    set((state) => ({
      toasts: [
        ...state.toasts,
        {
          ...toast,
          id: crypto.randomUUID(),
        },
      ].slice(-5),
    })),
  dismissToast: (id) =>
    set((state) => ({
      toasts: state.toasts.filter((toast) => toast.id !== id),
    })),
}));
