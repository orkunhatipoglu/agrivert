# Agrivert — API Routes Plan

This document plans the backend routes for the plant disease monitoring app,
based on the workflow: farmer uploads a plant photo → backend preprocesses it →
image is run through a fine-tuned CNN (MobileNetV2 or ResNet50, trained on
PlantVillage) → verdict is refined → verdict is returned to the frontend.

Before the route list, a few gaps in the stated workflow are worth fixing
first, since they change what routes are actually needed.

## Gaps in the stated workflow

> **Reconstructed.** The route tables below reference these by number, but the
> list itself was missing from this file — it lived only in the original
> planning conversation. What follows is rebuilt from the numbered references
> (#2, #4, #5, #6, #7, #9 are cited in the tables) and from what the
> implementation actually does. Treat the wording as a faithful summary, not
> the original text.

1. **Nobody owns a diagnosis.** "Farmer uploads a photo" assumes an identified
   user, but the workflow never establishes one. Without an owner there is no
   history, no privacy boundary, and no way to scope anything.
   → the `/auth` routes, and `owner_uid` on every record.

2. **The upload is assumed to be a usable photo.** It might be a PDF, a
   screenshot, a 40 MB burst, or a truncated JPEG that decodes halfway. Failing
   deep inside inference gives the farmer a useless error minutes later.
   → validation at `POST /diagnoses`, with an explicit `rejected` status.

3. **Nothing is persisted.** A verdict computed and returned is gone; the same
   leaf photographed next week can't be compared to this one.
   → stored diagnoses, `GET /diagnoses`, and the stored image route.

4. **Inference doesn't fit in a request.** Preprocessing plus a forward pass
   takes seconds, and holding an HTTP connection open for it fails badly on a
   phone on rural data.
   → the async job model: `POST` returns `queued`, then poll `GET
   /diagnoses/{id}` or subscribe to `/stream`.

5. **"Verdict is refined" is undefined.** A class label is not advice. A farmer
   reading `Tomato___Late_blight` still doesn't know what to do about it.
   → the disease knowledge base behind `/diseases`, and the `recommendation`
   field on a completed diagnosis.

6. **No spatial scoping.** A flat list of past diagnoses can't answer "is this
   spreading in the north field?", which is the actual question.
   → **Not addressed.** `/farms`, `/farms/{id}/plots` and the `farmId`/`plotId`
   filters were built and then removed; see "Farms & plots (removed)" below.
   History is scoped by owner, class and date only.

7. **The model is assumed correct.** It is ~65% accurate on real field photos,
   so it is wrong often, and nothing in the workflow lets a farmer say so.
   → `POST /diagnoses/{id}/feedback`, which doubles as the retraining corpus.

8. **Uncertainty has nowhere to go.** The workflow returns "a verdict" as if
   there is always one. A confident wrong diagnosis is worse for a farmer than
   an honest "I can't tell from this photo".
   → the `uncertain` status, a calibrated confidence threshold, and withholding
   the label below it.

9. **Verdicts aren't attributable.** When a model is retrained, past diagnoses
   become uninterpretable if nothing records which model produced them.
   → `modelVersion` on every completed diagnosis, plus the `/admin/models`
   routes.

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

### Farms & plots (removed)

`/farms` and `/farms/{farmId}/plots` implemented full CRUD with cascade delete
and ownership scoping, and diagnoses carried optional `farmId`/`plotId` tags.
All of it has been removed — routes, schemas, repositories, the `/farms` page
and the tag fields on a diagnosis.

Flaw #6 is therefore an open gap rather than a solved one. Anything that
reintroduces spatial scoping starts from that flaw, not from this table.

Note that removing the code does not remove data: existing `farms` and `plots`
Firestore documents, and `farm_id`/`plot_id` fields on diagnoses written
before the removal, are still there and are now simply unread.

### Diagnoses (the core photo → verdict workflow)

Modeled as an async job, per flaw #4.

| Method | Path | Description |
|---|---|---|
| POST | `/diagnoses` | Upload a photo (multipart). Runs validation (flaw #2), enqueues preprocessing + inference, returns `{ diagnosisId, status: "queued" }` |
| GET | `/diagnoses/{id}` | Poll status/result: `queued` \| `processing` \| `rejected` \| `uncertain` \| `completed` \| `failed`, plus verdict, confidence, and recommendation once completed |
| GET | `/diagnoses/{id}/stream` | WebSocket/SSE endpoint to push status updates instead of polling |
| GET | `/diagnoses/{id}/image` | Fetch the original uploaded photo |
| GET | `/diagnoses` | List diagnosis history, filterable by `dateRange`, `diseaseId`, `status` |
| DELETE | `/diagnoses/{id}` | Delete a diagnosis record and its stored image |
| POST | `/diagnoses/{id}/feedback` | Farmer confirms or corrects the verdict (flaw #7) — feeds a retraining dataset |

### Disease knowledge base

Backs the "refined verdict" with actual guidance, per flaw #5.

| Method | Path | Description |
|---|---|---|
| GET | `/diseases` | List all diseases the model can detect, with crop associations |
| GET | `/diseases/{id}` | Disease details: description, symptoms, severity, recommended treatment |

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
