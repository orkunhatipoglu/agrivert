/**
 * Wire types, mirrored from the FastAPI schemas in `backend/app/schemas/`.
 *
 * Response bodies are camelCase (pydantic `alias_generator=to_camel`), but
 * query and multipart form parameters are snake_case — FastAPI derives those
 * from the Python parameter names, which were never aliased. `lib/api.ts`
 * is the only place that has to care about the difference.
 */

export type DiagnosisStatus =
  | "queued"
  | "processing"
  | "rejected"
  | "uncertain"
  | "completed"
  | "failed"

/** Statuses the worker will never move away from. */
export const TERMINAL_STATUSES: readonly DiagnosisStatus[] = [
  "rejected",
  "uncertain",
  "completed",
  "failed",
]

export function isTerminal(status: DiagnosisStatus): boolean {
  return TERMINAL_STATUSES.includes(status)
}

export interface Alternative {
  rawLabel: string
  crop: string
  condition: string
  confidence: number
}

export interface DiagnosisCreated {
  diagnosisId: string
  status: DiagnosisStatus
}

export interface Diagnosis {
  diagnosisId: string
  status: DiagnosisStatus
  createdAt: string
  updatedAt?: string | null
  plotId?: string | null
  farmId?: string | null

  /**
   * Null while queued/processing — and deliberately null when `uncertain`
   * too. The backend withholds the verdict below the confidence threshold
   * rather than surfacing a guess.
   */
  crop?: string | null
  condition?: string | null
  healthy?: boolean | null
  rawLabel?: string | null
  diseaseId?: string | null

  confidence?: number | null
  threshold?: number | null
  /** False => the model never saw a real field photo of this class. */
  fieldValidated?: boolean | null
  alternatives: Alternative[]

  modelVersion?: string | null
  recommendation?: string | null

  /** Set when status is `rejected` or `failed`. */
  error?: string | null
}

export interface DiagnosisListItem {
  diagnosisId: string
  status: DiagnosisStatus
  createdAt: string
  crop?: string | null
  condition?: string | null
  healthy?: boolean | null
  confidence?: number | null
  plotId?: string | null
}

export interface DiagnosisList {
  items: DiagnosisListItem[]
  nextPageToken?: string | null
}

export interface FeedbackRequest {
  agrees: boolean
  /** Required when `agrees` is false. A model class, or `"unknown"`. */
  correctedRawLabel?: string | null
  note?: string | null
}

export interface FeedbackResponse {
  diagnosisId: string
  recorded: boolean
  agrees: boolean
  correctedRawLabel?: string | null
}

export interface UserProfile {
  uid: string
  email?: string | null
  displayName?: string | null
  emailVerified: boolean
  disabled: boolean
  isAdmin: boolean
  createdAt?: string | null
}

export type Severity = "unknown" | "low" | "moderate" | "high"

export interface DiseaseSummary {
  diseaseId: string
  rawLabel: string
  crop: string
  condition: string
  healthy: boolean
  fieldValidated: boolean
}

export interface Disease extends DiseaseSummary {
  description?: string | null
  symptoms: string[]
  treatment: string[]
  prevention: string[]
  severity: Severity
  references: string[]
  /** True once a human has written and reviewed the agronomic content. */
  contentReviewed: boolean
}

export interface DiseaseList {
  items: DiseaseSummary[]
}

export interface GeoPoint {
  latitude: number
  longitude: number
}

export interface Farm {
  farmId: string
  ownerUid: string
  name: string
  location?: GeoPoint | null
  region?: string | null
  createdAt: string
  updatedAt?: string | null
}

export interface Plot {
  plotId: string
  farmId: string
  name: string
  cropType: string
  areaHectares?: number | null
  location?: GeoPoint | null
  createdAt: string
  updatedAt?: string | null
}

export interface FarmList {
  items: Farm[]
}

export interface PlotList {
  items: Plot[]
}

export interface ModelVersionInfo {
  version: string
  modelName?: string | null
  architecture?: string | null
  numClasses?: number | null
  bestEpoch?: number | null
  active: boolean
  registeredAt?: string | null
  /**
   * Straight from the version's metadata.json. `test_field` is the number to
   * judge on; `test_studio` is inflated by the studio-photo domain gap.
   */
  metrics: Record<string, number | string | null>
  confidenceThreshold?: number | null
  temperature?: number | null
  caveat?: string | null
}

export interface ModelVersionList {
  items: ModelVersionInfo[]
  activeVersion?: string | null
}

export interface ActivateResponse {
  version: string
  active: boolean
  detail: string
}

export interface FeedbackStats {
  totalDiagnoses: number
  completed: number
  uncertain: number
  rejected: number
  failed: number
  feedbackCount: number
  agreed: number
  corrected: number
  /** Not accuracy — only responders are counted, and they skew negative. */
  agreementRateOfResponders?: number | null
}

export interface AdminStats {
  activeModelVersion?: string | null
  diagnoses: FeedbackStats
}

export interface HealthResponse {
  status: string
  environment: string
  modelVersion?: string | null
  modelReady: boolean
  firestoreReady: boolean
  brokerReady: boolean
  detail?: string | null
}

export interface AppNotification {
  notificationId: string
  kind: string
  title: string
  body?: string | null
  read: boolean
}

export interface NotificationList {
  items: AppNotification[]
}

export interface DiagnosisFilters {
  farmId?: string
  plotId?: string
  diseaseId?: string
  status?: DiagnosisStatus
  dateFrom?: string
  dateTo?: string
  limit?: number
}
