# Agrivert — API Routes Plan

This document plans the backend routes for the plant disease monitoring app,
based on the workflow: farmer uploads a plant photo → backend preprocesses it →
image is run through a fine-tuned CNN (MobileNetV2 or ResNet50, trained on
PlantVillage) → verdict is refined → verdict is returned to the frontend.

Before the route list, a few gaps in the stated workflow are worth fixing
first, since they change what routes are actually needed.


## Proposed routes

Base path: `/api/v1`

### Auth

| Method | Path | Description |
|---|---|---|
| POST | `/auth/register` | Create a farmer account |
| POST | `/auth/login` | Authenticate, issue access/refresh tokens |
| POST | `/auth/refresh` | Refresh an access token |
| POST | `/auth/logout` | Invalidate refresh token |
| GET | `/auth/me` | Current user profile |

### Farms & plots

Scoping diagnoses to a farm/plot is what makes history and trend-tracking
possible (see flaw #6).

| Method | Path | Description |
|---|---|---|
| GET | `/farms` | List the current user's farms |
| POST | `/farms` | Create a farm |
| GET | `/farms/{farmId}` | Farm details |
| PATCH | `/farms/{farmId}` | Update farm (name, location) |
| DELETE | `/farms/{farmId}` | Delete a farm |
| GET | `/farms/{farmId}/plots` | List plots/fields within a farm |
| POST | `/farms/{farmId}/plots` | Create a plot (crop type, area, location) |
| PATCH | `/farms/{farmId}/plots/{plotId}` | Update a plot |
| DELETE | `/farms/{farmId}/plots/{plotId}` | Delete a plot |

### Diagnoses (the core photo → verdict workflow)

Modeled as an async job, per flaw #4.

| Method | Path | Description |
|---|---|---|
| POST | `/diagnoses` | Upload a photo (multipart), optionally with `plotId`. Runs validation (flaw #2), enqueues preprocessing + inference, returns `{ diagnosisId, status: "queued" }` |
| GET | `/diagnoses/{id}` | Poll status/result: `queued` \| `processing` \| `rejected` \| `uncertain` \| `completed` \| `failed`, plus verdict, confidence, and recommendation once completed |
| GET | `/diagnoses/{id}/stream` | WebSocket/SSE endpoint to push status updates instead of polling |
| GET | `/diagnoses/{id}/image` | Fetch the original uploaded photo |
| GET | `/diagnoses` | List diagnosis history, filterable by `farmId`, `plotId`, `dateRange`, `diseaseId`, `status` |
| DELETE | `/diagnoses/{id}` | Delete a diagnosis record and its stored image |
| POST | `/diagnoses/{id}/feedback` | Farmer confirms or corrects the verdict (flaw #7) — feeds a retraining dataset |

### Disease knowledge base

Backs the "refined verdict" with actual guidance, per flaw #5.

| Method | Path | Description |
|---|---|---|
| GET | `/diseases` | List all diseases the model can detect, with crop associations |
| GET | `/diseases/{id}` | Disease details: description, symptoms, severity, recommended treatment |

### Notifications (optional, v1.1+)

| Method | Path | Description |
|---|---|---|
| GET | `/notifications` | List alerts (e.g. disease outbreak trending in the farmer's region) |
| POST | `/notifications/subscribe` | Opt into regional outbreak alerts |

### Admin / model management

Supports flaw #9 (model versioning and auditability). Restricted to an
admin/internal role.

| Method | Path | Description |
|---|---|---|
| GET | `/admin/models` | List registered model versions and their eval metrics |
| POST | `/admin/models/{version}/activate` | Promote a model version to production |
| GET | `/admin/stats` | Aggregate accuracy/feedback stats across diagnoses |

### Misc

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness/readiness check |

---

## Notes on the inference step specifically

- Preprocessing (resize/normalize/orientation-fix) and inference are
  probably worth running as an internal service or background worker, not
  inline in the request handler for `POST /diagnoses` — this is what makes
  the async job model in the Diagnoses section work, and lets the
  ML service scale independently of the API.
- `GET /diagnoses/{id}` should always return a `modelVersion` field once
  completed, and the confidence score alongside the label, not just the
  top-1 class — so the frontend can render "uncertain" states honestly
  instead of a false-confidence diagnosis.
