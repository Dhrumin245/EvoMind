import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from './client';
import type {
  AgentInfoResponse,
  AgentQuery,
  AgentResponse,
  AuthSessionResponse,
  ApiKeyCreateRequest,
  ApiKeyCreateResponse,
  ApiKeyDeleteResponse,
  ApiKeyListResponse,
  BatchAgentQuery,
  BatchAgentResponse,
  BillingAccountResponse,
  BillingLedgerResponse,
  BillingTiersResponse,
  BillingTopupConfirmRequest,
  BillingTopupConfirmResponse,
  BillingTopupResponse,
  CheckpointListResponse,
  CreateCheckpointResponse,
  GenomeListResponse,
  GenomeSummary,
  GenomeType,
  HealthCheck,
  JobEventListResponse,
  JobListResponse,
  JobSummary,
  MetricsResponse,
  ReadinessCheck,
  TenantLimitsResponse,
  TrainStatus,
  UsageExportResponse,
  UsageSummaryResponse,
  WebhookCreateRequest,
  WebhookDeleteResponse,
  WebhookDeliveryListResponse,
  WebhookListResponse,
  WebhookSummary,
} from './types';

export const queryKeys = {
  currentUser: ['currentUser'] as const,
  health: ['health'] as const,
  readiness: ['readiness'] as const,
  trainStatus: (jobId = '') => ['trainStatus', jobId] as const,
  trainInsights: (jobId = '') => ['trainInsights', jobId] as const,
  jobs: ['jobs'] as const,
  jobEvents: (jobId = '') => ['jobEvents', jobId] as const,
  metrics: (jobId = '') => ['metrics', jobId] as const,
  genomes: (jobId = '', genomeType?: GenomeType) => ['genomes', jobId, genomeType] as const,
  genomeDetail: (jobId = '', genomeId = '', genomeType?: GenomeType) => ['genomeDetail', jobId, genomeId, genomeType] as const,
  checkpoints: (jobId = '') => ['checkpoints', jobId] as const,
  agentInfo: (jobId = '', genomeType: GenomeType = 'prey', generation?: number | null) => ['agentInfo', jobId, genomeType, generation] as const,
  apiKeys: ['apiKeys'] as const,
  billingAccount: ['billingAccount'] as const,
  billingLedger: ['billingLedger'] as const,
  billingTiers: ['billingTiers'] as const,
  usageLimits: ['usageLimits'] as const,
  usageSummary: ['usageSummary'] as const,
  usageExport: (days = 7, limit = 100) => ['usageExport', days, limit] as const,
  webhooks: ['webhooks'] as const,
  webhookDeliveries: (webhookId = '', status?: string) => ['webhookDeliveries', webhookId, status] as const,
};

export function useLogin() {
  return useMutation({
    mutationFn: async (payload: { email: string; password: string }) =>
      (await apiClient.post<AuthSessionResponse>('/auth/login', payload)).data,
  });
}

export function useRegister() {
  return useMutation({
    mutationFn: async (payload: { email: string; password: string; tenant_id?: string; name?: string }) =>
      (await apiClient.post<AuthSessionResponse>('/auth/register', payload)).data,
  });
}

export function useLogout() {
  return useMutation({
    mutationFn: async () => (await apiClient.post<{ logged_out: boolean }>('/auth/logout')).data,
  });
}

export function useHealth(refetchInterval = 15000) {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: async () => (await apiClient.get<HealthCheck>('/health')).data,
    refetchInterval,
  });
}

export function useReadiness(refetchInterval = 15000) {
  return useQuery({
    queryKey: queryKeys.readiness,
    queryFn: async () =>
      (await apiClient.get<ReadinessCheck>('/health/readiness', { validateStatus: () => true })).data,
    refetchInterval,
  });
}

export function useJobs() {
  return useQuery({
    queryKey: queryKeys.jobs,
    queryFn: async () => (await apiClient.get<JobListResponse>('/jobs')).data,
    staleTime: 5000,
  });
}

export function useTrainStatus(jobId = '', refetchInterval = 7500) {
  return useQuery({
    queryKey: queryKeys.trainStatus(jobId),
    queryFn: async () => (await apiClient.get<TrainStatus>(`/jobs/${jobId}/train/status`)).data,
    enabled: Boolean(jobId),
    refetchInterval,
  });
}

export function useTrainInsights(jobId = '', lastN = 10, refetchInterval = 15000) {
  return useQuery({
    queryKey: queryKeys.trainInsights(jobId),
    queryFn: async () => (await apiClient.get<Record<string, unknown>>(`/jobs/${jobId}/train/insights`, { params: { last_n: lastN } })).data,
    enabled: Boolean(jobId),
    refetchInterval,
  });
}

export function useMetrics(jobId = '', refetchInterval = 7500) {
  return useQuery({
    queryKey: queryKeys.metrics(jobId),
    queryFn: async () =>
      (await apiClient.get<MetricsResponse>(`/jobs/${jobId}/train/metrics`, { params: { limit: 100 } })).data,
    enabled: Boolean(jobId),
    refetchInterval,
  });
}

export function useJobEvents(jobId = '', refetchInterval = 7500) {
  return useQuery({
    queryKey: queryKeys.jobEvents(jobId),
    queryFn: async () =>
      (await apiClient.get<JobEventListResponse>(`/jobs/${jobId}/events`, { params: { limit: 50 } })).data,
    enabled: Boolean(jobId),
    refetchInterval,
  });
}

export function useGenomes(jobId = '', genomeType?: GenomeType) {
  return useQuery({
    queryKey: queryKeys.genomes(jobId, genomeType),
    queryFn: async () =>
      (await apiClient.get<GenomeListResponse>(`/jobs/${jobId}/genomes`, { params: { genome_type: genomeType } })).data,
    enabled: Boolean(jobId),
  });
}

export function useGenomeDetail(jobId = '', genomeId?: string, genomeType?: GenomeType) {
  return useQuery({
    queryKey: queryKeys.genomeDetail(jobId, genomeId || '', genomeType),
    queryFn: async () =>
      (await apiClient.get<GenomeSummary>(`/jobs/${jobId}/genomes/${genomeId}`, { params: { genome_type: genomeType } })).data,
    enabled: Boolean(jobId && genomeId),
  });
}

export function useCheckpoints(jobId = '', limit = 50) {
  return useQuery({
    queryKey: queryKeys.checkpoints(jobId),
    queryFn: async () =>
      (await apiClient.get<CheckpointListResponse>(`/jobs/${jobId}/train/checkpoints`, { params: { limit } })).data,
    enabled: Boolean(jobId),
  });
}

export function useAgentInfo(jobId = '', genomeType: GenomeType = 'prey', generation?: number | null) {
  return useQuery({
    queryKey: queryKeys.agentInfo(jobId, genomeType, generation),
    queryFn: async () =>
      (await apiClient.get<AgentInfoResponse>(`/jobs/${jobId}/agent/info`, { params: { genome_type: genomeType, generation: generation || undefined } })).data,
    enabled: Boolean(jobId),
  });
}

export function useApiKeys() {
  return useQuery({
    queryKey: queryKeys.apiKeys,
    queryFn: async () => (await apiClient.get<ApiKeyListResponse>('/auth/keys')).data,
  });
}

export function useBillingAccount() {
  return useQuery({
    queryKey: queryKeys.billingAccount,
    queryFn: async () => (await apiClient.get<BillingAccountResponse>('/billing/account')).data,
  });
}

export function useBillingLedger() {
  return useQuery({
    queryKey: queryKeys.billingLedger,
    queryFn: async () => (await apiClient.get<BillingLedgerResponse>('/billing/ledger')).data,
  });
}

export function useUsageLimits() {
  return useQuery({
    queryKey: queryKeys.usageLimits,
    queryFn: async () => (await apiClient.get<TenantLimitsResponse>('/usage/limits')).data,
  });
}

export function useUsageSummary(refetchInterval = 15000) {
  return useQuery({
    queryKey: queryKeys.usageSummary,
    queryFn: async () => (await apiClient.get<UsageSummaryResponse>('/usage/summary')).data,
    refetchInterval,
  });
}

export function useBillingTiers() {
  return useQuery({
    queryKey: queryKeys.billingTiers,
    queryFn: async () => (await apiClient.get<BillingTiersResponse>('/usage/billing-tiers')).data,
  });
}

export function useUsageExport(days = 7, limit = 100) {
  return useQuery({
    queryKey: queryKeys.usageExport(days, limit),
    queryFn: async () =>
      (await apiClient.get<UsageExportResponse>('/usage/export', { params: { format: 'json', days, limit } })).data,
  });
}

export function useCreateJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (payload: { job_id?: string; name?: string }) =>
      (await apiClient.post<JobSummary>('/jobs', payload)).data,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.jobs }),
  });
}

export function useJobCommand(jobId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (input: 'start' | 'stop' | { command: 'resume'; checkpoint_path: string }) => {
      const command = typeof input === 'string' ? input : input.command;
      const path = command === 'resume' ? `/jobs/${jobId}/train/resume` : `/jobs/${jobId}/train/${command}`;
      const body = command === 'resume' && typeof input !== 'string' ? { checkpoint_path: input.checkpoint_path } : undefined;
      return (await apiClient.post<TrainStatus>(path, body)).data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.trainStatus(jobId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.jobs });
      queryClient.invalidateQueries({ queryKey: queryKeys.jobEvents(jobId) });
    },
  });
}

export function useCreateCheckpoint(jobId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (path?: string) =>
      (await apiClient.post<CreateCheckpointResponse>(`/jobs/${jobId}/train/checkpoints`, path ? { path } : {})).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.jobEvents(jobId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.checkpoints(jobId) });
    },
  });
}

export function useAgentAction(jobId: string) {
  return useMutation({
    mutationFn: async (payload: AgentQuery) =>
      (await apiClient.post<AgentResponse>(`/jobs/${jobId}/agent/action`, payload)).data,
  });
}

export function useBatchAgentAction(jobId: string) {
  return useMutation({
    mutationFn: async (payload: BatchAgentQuery) =>
      (await apiClient.post<BatchAgentResponse>(`/jobs/${jobId}/agent/action/batch`, payload)).data,
  });
}

export function useCreateApiKey() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (payload: ApiKeyCreateRequest) =>
      (await apiClient.post<ApiKeyCreateResponse>('/auth/keys', payload)).data,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.apiKeys }),
  });
}

export function useDeleteApiKey() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (keyId: string) => (await apiClient.delete<ApiKeyDeleteResponse>(`/auth/keys/${keyId}`)).data,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.apiKeys }),
  });
}

export function useCreateTopup() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (payload: { amount_inr: number; description?: string }) =>
      (await apiClient.post<BillingTopupResponse>('/billing/topups', payload)).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.billingAccount });
      queryClient.invalidateQueries({ queryKey: queryKeys.billingLedger });
    },
  });
}

export function useConfirmTopup() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (payload: BillingTopupConfirmRequest) =>
      (await apiClient.post<BillingTopupConfirmResponse>('/billing/topups/confirm', payload)).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.billingAccount });
      queryClient.invalidateQueries({ queryKey: queryKeys.billingLedger });
    },
  });
}

export function useWebhooks() {
  return useQuery({
    queryKey: queryKeys.webhooks,
    queryFn: async () => (await apiClient.get<WebhookListResponse>('/webhooks')).data,
  });
}

export function useCreateWebhook() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (payload: WebhookCreateRequest) =>
      (await apiClient.post<WebhookSummary>('/webhooks', payload)).data,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.webhooks }),
  });
}

export function useDeleteWebhook() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (webhookId: string) =>
      (await apiClient.delete<WebhookDeleteResponse>(`/webhooks/${webhookId}`)).data,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.webhooks }),
  });
}

export function useWebhookDeliveries(webhookId?: string, status?: string) {
  return useQuery({
    queryKey: queryKeys.webhookDeliveries(webhookId || '', status),
    queryFn: async () =>
      (await apiClient.get<WebhookDeliveryListResponse>(`/webhooks/${webhookId}/deliveries`, { params: { limit: 50, status: status || undefined } })).data,
    enabled: Boolean(webhookId),
  });
}
