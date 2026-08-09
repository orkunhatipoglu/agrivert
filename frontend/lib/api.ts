/**
 * Typed client for the Agrivert API.
 *
 * Every route group in `backend/app/routers/` is represented here, including
 * the ones that currently answer 501. Those throw `NotImplementedError` so a
 * page can say plainly that the backend hasn't shipped it yet, rather than
 * rendering an empty list that reads as "you have no diagnoses".
 */

import { getIdToken } from "@/lib/firebase"
import type {
  ActivateResponse,
  AdminStats,
  Diagnosis,
  DiagnosisCreated,
  DiagnosisFilters,
  DiagnosisList,
  Disease,
  DiseaseList,
  FeedbackRequest,
  FeedbackResponse,
  HealthResponse,
  ModelVersionList,
  UserProfile,
} from "@/lib/types"

export const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "")

export const API_PREFIX = process.env.NEXT_PUBLIC_API_PREFIX ?? "/api/v1"

const ROOT = `${API_BASE}${API_PREFIX}`

export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown

  constructor(status: number, message: string, detail: unknown) {
    super(message)
    this.name = "ApiError"
    this.status = status
    this.detail = detail
  }

  /** The route exists and is authorised, but the handler is a stub. */
  get isNotImplemented(): boolean {
    return this.status === 501
  }

  get isUnauthorized(): boolean {
    return this.status === 401
  }
}

/** Shape of the 422 body from `POST /diagnoses` when validation rejects a photo. */
export interface RejectedUpload {
  diagnosisId: string
  status: "rejected"
  reason: string
}

export function asRejectedUpload(error: unknown): RejectedUpload | null {
  if (!(error instanceof ApiError) || error.status !== 422) return null
  const detail = error.detail
  if (
    detail &&
    typeof detail === "object" &&
    "reason" in detail &&
    "diagnosisId" in detail
  ) {
    return detail as RejectedUpload
  }
  return null
}

/**
 * FastAPI puts a string in `detail` for HTTPException, an object for the
 * rejected-upload case, and an array of error objects for request-validation
 * failures. Flatten all three into something a toast can show.
 */
function messageFromDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) =>
        item && typeof item === "object" && "msg" in item
          ? String((item as { msg: unknown }).msg)
          : null
      )
      .filter(Boolean)
    if (parts.length) return parts.join("; ")
  }
  if (detail && typeof detail === "object" && "reason" in detail) {
    return String((detail as { reason: unknown }).reason)
  }
  return fallback
}

async function authHeaders(): Promise<Record<string, string>> {
  const token = await getIdToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function raise(response: Response): Promise<never> {
  let detail: unknown = null
  try {
    const body = await response.json()
    detail = body?.detail ?? body
  } catch {
    detail = null
  }
  throw new ApiError(
    response.status,
    messageFromDetail(detail, `${response.status} ${response.statusText}`),
    detail
  )
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${ROOT}${path}`, {
    ...init,
    headers: {
      ...(await authHeaders()),
      ...(init.headers ?? {}),
    },
  })
  if (!response.ok) await raise(response)
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

function jsonBody(body: unknown): RequestInit {
  return {
    body: JSON.stringify(body),
    headers: { "Content-Type": "application/json" },
  }
}

/** Drops undefined/empty values so we never send `?disease_id=undefined`. */
function query(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === "") continue
    search.set(key, String(value))
  }
  const qs = search.toString()
  return qs ? `?${qs}` : ""
}

// --- Auth -----------------------------------------------------------------

export const authApi = {
  /**
   * Creates the Firebase Auth user server-side. The client then signs in
   * separately to obtain tokens — the API never issues them.
   */
  register: (body: {
    email: string
    password: string
    displayName?: string
  }): Promise<UserProfile> =>
    request<UserProfile>("/auth/register", { method: "POST", ...jsonBody(body) }),

  me: (): Promise<UserProfile> => request<UserProfile>("/auth/me"),

  /** 501 by design — the Firebase client SDK signs in. Kept for completeness. */
  login: (body: { email: string; password: string }): Promise<never> =>
    request<never>("/auth/login", { method: "POST", ...jsonBody(body) }),

  /** 501 by design — `getIdToken(true)` refreshes client-side. */
  refresh: (refreshToken: string): Promise<never> =>
    request<never>("/auth/refresh", {
      method: "POST",
      ...jsonBody({ refreshToken }),
    }),

  /** 501 by design — server-side revocation isn't wired up yet. */
  logout: (): Promise<void> => request<void>("/auth/logout", { method: "POST" }),
}

// --- Diagnoses ------------------------------------------------------------

export const diagnosesApi = {
  create: async (input: { file: File }): Promise<DiagnosisCreated> => {
    const form = new FormData()
    form.append("file", input.file)
    return request<DiagnosisCreated>("/diagnoses", {
      method: "POST",
      body: form,
    })
  },

  list: (filters: DiagnosisFilters = {}): Promise<DiagnosisList> =>
    request<DiagnosisList>(
      `/diagnoses${query({
        disease_id: filters.diseaseId,
        status: filters.status,
        date_from: filters.dateFrom,
        date_to: filters.dateTo,
        limit: filters.limit,
      })}`
    ),

  get: (id: string): Promise<Diagnosis> =>
    request<Diagnosis>(`/diagnoses/${encodeURIComponent(id)}`),

  remove: (id: string): Promise<void> =>
    request<void>(`/diagnoses/${encodeURIComponent(id)}`, { method: "DELETE" }),

  feedback: (id: string, body: FeedbackRequest): Promise<FeedbackResponse> =>
    request<FeedbackResponse>(`/diagnoses/${encodeURIComponent(id)}/feedback`, {
      method: "POST",
      ...jsonBody(body),
    }),

  /**
   * The stored photo. Fetched as a blob rather than pointed at with an <img
   * src> because the route requires a bearer token, which an <img> cannot send.
   */
  image: async (id: string): Promise<Blob> => {
    const response = await fetch(
      `${ROOT}/diagnoses/${encodeURIComponent(id)}/image`,
      { headers: await authHeaders() }
    )
    if (!response.ok) await raise(response)
    return response.blob()
  },
}

/**
 * Subscribe to `GET /diagnoses/{id}/stream`.
 *
 * `EventSource` can't carry an Authorization header, so this reads the SSE
 * body off `fetch` and parses the frames by hand. Returns an abort function.
 */
export function streamDiagnosis(
  id: string,
  handlers: {
    onStatus: (diagnosis: Diagnosis) => void
    onError?: (error: Error) => void
    onClose?: () => void
  }
): () => void {
  const controller = new AbortController()

  void (async () => {
    try {
      const response = await fetch(
        `${ROOT}/diagnoses/${encodeURIComponent(id)}/stream`,
        {
          headers: { ...(await authHeaders()), Accept: "text/event-stream" },
          signal: controller.signal,
        }
      )
      if (!response.ok) await raise(response)
      if (!response.body) throw new Error("stream response had no body")

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // Frames are separated by a blank line; keep the trailing partial.
        const frames = buffer.split(/\r?\n\r?\n/)
        buffer = frames.pop() ?? ""

        for (const frame of frames) {
          let event = "message"
          const data: string[] = []
          for (const line of frame.split(/\r?\n/)) {
            if (line.startsWith("event:")) event = line.slice(6).trim()
            else if (line.startsWith("data:")) data.push(line.slice(5).trim())
          }
          if (!data.length) continue

          const payload = data.join("\n")
          if (event === "status") {
            handlers.onStatus(JSON.parse(payload) as Diagnosis)
          } else if (event === "error" || event === "timeout") {
            handlers.onError?.(
              new Error(messageFromDetail(JSON.parse(payload)?.detail, event))
            )
          }
        }
      }
      handlers.onClose?.()
    } catch (error) {
      if (controller.signal.aborted) return
      handlers.onError?.(
        error instanceof Error ? error : new Error(String(error))
      )
    }
  })()

  return () => controller.abort()
}

// --- Diseases -------------------------------------------------------------

export const diseasesApi = {
  list: (): Promise<DiseaseList> => request<DiseaseList>("/diseases"),
  get: (id: string): Promise<Disease> =>
    request<Disease>(`/diseases/${encodeURIComponent(id)}`),
}

// --- Admin ----------------------------------------------------------------

export const adminApi = {
  models: (): Promise<ModelVersionList> =>
    request<ModelVersionList>("/admin/models"),
  activate: (version: string): Promise<ActivateResponse> =>
    request<ActivateResponse>(
      `/admin/models/${encodeURIComponent(version)}/activate`,
      { method: "POST" }
    ),
  /** 501 — Firestore has no GROUP BY, so this needs a rollup job first. */
  stats: (): Promise<AdminStats> => request<AdminStats>("/admin/stats"),
}

// --- Misc -----------------------------------------------------------------

/** Unauthenticated, so it also works as a "can the browser reach the API" probe. */
export const healthApi = {
  get: async (): Promise<HealthResponse> => {
    const response = await fetch(`${ROOT}/health`)
    if (!response.ok) await raise(response)
    return (await response.json()) as HealthResponse
  },
}
