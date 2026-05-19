export type TrainStatusEnum = 'queued' | 'stopped' | 'running' | 'paused' | 'error';

export interface FitnessPair {
  best: number;
  average: number;
}

export interface FitnessStatus {
  prey: FitnessPair;
  predator: FitnessPair;
}

export interface LearningStatus {
  adaptability: number;
  meta_effectiveness: number;
  performance_change: number;
  instability: number;
}

export interface BehaviorStatus {
  success_rate: number;
  stability: number;
  novelty: number;
}

export interface DiversityStatus {
  prey_species: number;
  predator_species: number;
}

export interface NeuralHealthStatus {
  dead_connections: number;
  saturation: number;
}

export interface SystemStatus {
  evaluation_time_sec: number;
  status: TrainStatusEnum;
  uptime_seconds: number;
  last_update: string;
}

export interface TrainStatus {
  generation: number;
  stage: string;
  fitness: FitnessStatus;
  learning: LearningStatus;
  behavior: BehaviorStatus;
  diversity: DiversityStatus;
  neural_health: NeuralHealthStatus;
  system: SystemStatus;
  status: TrainStatusEnum;
  best_prey_fitness: number;
  best_predator_fitness: number;
  mean_prey_fitness: number;
  mean_predator_fitness: number;
  curriculum_stage: string;
  total_generations_trained: number;
  uptime_seconds: number;
  last_update: string;
}

export interface JobSummary {
  job_id: string;
  tenant_id: string;
  name: string;
  base_dir: string;
  created_at: string;
  updated_at: string;
  status: string;
  generation: number;
}

export interface ListResponse<T> {
  count: number;
  items: T[];
}

export type JobListResponse = ListResponse<JobSummary>;

export interface JobEvent {
  event_id: string;
  tenant_id: string;
  job_id: string;
  event_type: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export type JobEventListResponse = ListResponse<JobEvent>;

export interface MetricsResponse {
  source: string;
  total: number;
  count: number;
  limit: number | null;
  offset: number;
  since_generation: number | null;
  items: MetricRow[];
}

export interface HealthCheck {
  status: 'healthy' | 'warning' | 'error';
  message: string;
  uptime_seconds: number;
}

export interface ReadinessComponent {
  name: string;
  healthy: boolean;
  detail: string;
}

export interface ReadinessCheck {
  status: 'ready' | 'not_ready';
  message: string;
  uptime_seconds: number;
  components: ReadinessComponent[];
}

export interface MetricRow {
  generation?: number;
  best_prey_fitness?: number;
  mean_prey_fitness?: number;
  best_predator_fitness?: number;
  mean_predator_fitness?: number;
  prey_species?: number;
  predator_species?: number;
  adaptability?: number;
  meta_effectiveness?: number;
  performance_change?: number;
  instability?: number;
  success_rate?: number;
  stability?: number;
  novelty?: number;
  dead_connections?: number;
  saturation?: number;
  [key: string]: unknown;
}

export type GenomeType = 'prey' | 'predator';

export interface GenomeSummary {
  genome_id: string;
  genome_type: GenomeType;
  fitness: number;
  generation: number;
  source: string;
  gene_count: number;
  input_size: number;
  output_size: number;
  architecture: string | null;
}

export type GenomeListResponse = ListResponse<GenomeSummary>;

export interface CheckpointSummary {
  checkpoint_path: string;
  generation: number;
  saved_at_utc: string;
  config_path: string | null;
  experiment_path: string | null;
  metrics_path: string | null;
  marker_exists: boolean;
}

export type CheckpointListResponse = ListResponse<CheckpointSummary>;

export interface AgentQuery {
  observation: number[];
  genome_type: GenomeType;
  generation?: number | null;
  max_action_length?: number;
}

export interface AgentResponse {
  action: number[];
  genome_id: string;
  genome_fitness: number;
  genome_type: GenomeType;
  generation: number;
  confidence: number;
}

export interface BatchAgentQuery {
  observations: number[][];
  genome_type: GenomeType;
  generation?: number | null;
  max_action_length?: number;
}

export interface BatchAgentResponse {
  genome_id: string;
  genome_type: GenomeType;
  generation: number;
  genome_fitness: number;
  confidence: number;
  batch_size: number;
  actions: number[][];
}

export interface AgentInfoResponse {
  available: boolean;
  genome_id?: string;
  genome_type: GenomeType;
  fitness?: number;
  generation?: number | null;
  source?: string;
  gene_count?: number;
  input_size?: number;
  output_size?: number;
  architecture?: string | null;
}

export interface CreateCheckpointResponse {
  checkpoint: CheckpointSummary;
}

export interface ApiKeySummary {
  key_id: string;
  name: string;
  tenant_id: string;
  status: string;
  role: string;
  scopes: string[];
  created_at: string | null;
  updated_at: string | null;
  last_used_at: string | null;
  expires_at: string | null;
  revoked_at: string | null;
  expired_at: string | null;
  rotated_at: string | null;
  rotated_from_key_id: string | null;
  replaced_by_key_id: string | null;
}

export interface ApiKeyCreateRequest {
  name: string;
  role?: string;
  scopes?: string[];
  expires_at?: string | null;
}

export interface ApiKeyListResponse extends ListResponse<ApiKeySummary> {
  tenant_id: string;
}

export interface ApiKeyCreateResponse {
  key: ApiKeySummary;
  api_key: string;
}

export interface ApiKeyDeleteResponse {
  key_id: string;
  deleted: boolean;
}

export interface BillingAccountResponse {
  tenant_id: string;
  currency: string;
  available_credit_inr: number;
  outstanding_amount_inr: number;
  total_credited_inr: number;
  total_debited_inr: number;
  prepaid_required: boolean;
}

export interface BillingLedgerEntry {
  entry_id: number;
  tenant_id: string;
  entry_type: string;
  amount_inr: number;
  balance_after_inr: number;
  currency: string;
  description: string;
  reference_type: string | null;
  reference_id: string | null;
  created_at: string;
}

export interface BillingLedgerResponse extends ListResponse<BillingLedgerEntry> {
  tenant_id: string;
}

export interface BillingTopupResponse {
  tenant_id: string;
  topup_id: string;
  provider: string;
  status: string;
  amount_inr: number;
  currency: string;
  receipt: string;
  provider_order_id: string;
  checkout_key_id: string;
  created_at: string;
}

export interface BillingTopupConfirmRequest {
  razorpay_order_id: string;
  razorpay_payment_id: string;
  razorpay_signature: string;
}

export interface BillingTopupConfirmResponse {
  tenant_id: string;
  topup_id: string;
  provider: string;
  status: string;
  amount_inr: number;
  payment_id: string | null;
  balance_inr: number;
  credited: boolean;
}

export interface TenantLimitsResponse {
  tenant_id: string;
  requests_per_minute: number;
  requests_per_day: number;
  max_jobs: number;
}

export interface AuthUserResponse {
  user_id: string;
  email: string;
  name: string;
  tenant_id: string;
  role: string;
  scopes: string[];
  created_at: string;
  last_login_at: string | null;
}

export interface AuthSessionResponse {
  token: string;
  user: AuthUserResponse;
}

export interface UsageSummaryResponse {
  tenant_id: string;
  requests_last_minute: number;
  requests_last_day: number;
  requests_total: number;
  requests_per_minute_limit: number;
  requests_per_day_limit: number;
  max_jobs: number;
  remaining_this_minute: number;
  remaining_today: number;
  estimated_cost_last_day_inr: number;
  estimated_cost_total_inr: number;
}

export interface BillingTierInfo {
  method: string;
  route_template: string;
  billing_tier: string;
  unit_name: string;
  unit_price_inr: number;
  description: string;
}

export type BillingTiersResponse = ListResponse<BillingTierInfo>;

export interface UsageExportRow {
  created_at: string;
  tenant_id: string;
  key_id: string;
  method: string;
  path: string;
  route_template: string | null;
  status_code: number;
  duration_ms: number;
  job_id: string | null;
  billing_tier: string;
  billed_tokens: number;
  unit_price_inr: number;
  estimated_cost_inr: number;
}

export interface UsageExportResponse extends ListResponse<UsageExportRow> {
  tenant_id: string;
}

export interface WebhookCreateRequest {
  url: string;
  description?: string | null;
  subscribed_events?: string[];
  secret?: string | null;
}

export interface WebhookSummary {
  webhook_id: string;
  tenant_id: string;
  url: string;
  description: string;
  subscribed_events: string[];
  status: string;
  created_at: string;
  updated_at: string;
  last_delivery_at: string | null;
  last_delivery_status: string | null;
  last_delivery_error: string | null;
}

export type WebhookListResponse = ListResponse<WebhookSummary>;

export interface WebhookDeleteResponse {
  webhook_id: string;
  deleted: boolean;
}

export interface WebhookDeliveryAttempt {
  attempt_id: string;
  delivery_id: string;
  attempt_number: number;
  status: string;
  response_status_code: number | null;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface WebhookDelivery {
  delivery_id: string;
  webhook_id: string;
  event_id: string;
  tenant_id: string;
  job_id: string;
  event_type: string;
  status: string;
  attempt_count: number;
  max_attempts: number;
  next_retry_at: string | null;
  delivered_at: string | null;
  last_error: string | null;
  created_at: string;
  updated_at: string;
  attempts: WebhookDeliveryAttempt[];
}

export type WebhookDeliveryListResponse = ListResponse<WebhookDelivery>;
