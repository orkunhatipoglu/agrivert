# Agrivert Frontend

Next.js 16 (App Router) + Tailwind v4 + shadcn/ui. Talks to the FastAPI
backend in `../backend`, authenticating with Firebase ID tokens.

## Setup

```bash
cp .env.example .env.local     # fill in the API URL + Firebase web config
npm install
npm run dev                    # http://localhost:3000
```

The Firebase web app **must** belong to the same Firebase project as the
backend's service account, or the API rejects every token it is handed.

With the backend running (`docker compose -f ../backend/docker-compose.yml up`)
the sidebar status dot turns green once the model, Firestore and the Redis
broker all report ready.

## How auth works here

The API never issues tokens. `POST /auth/login`, `/auth/refresh` and
`/auth/logout` return 501 on purpose — with Firebase Auth the client SDK owns
the credential lifecycle. So:

- **Sign in** — `signInWithEmailAndPassword` in the browser
- **Register** — `POST /auth/register` (the admin SDK creates the user), then
  sign in client-side to get the token
- **Refresh** — `getIdToken()` handles it, which is why `lib/api.ts` fetches a
  fresh token per request rather than caching one
- **Sign out** — client-side `signOut`; the API's `check_revoked=True` still
  rejects revoked tokens

`hooks/use-auth.tsx` wraps all of it. `AuthGuard` redirects signed-out users,
but it is a convenience, not the security boundary — that is
`get_current_user` on the server.

## Route coverage

Every route in `../ROUTES.md` has a client method in `lib/api.ts` and a place
in the UI.

| Backend | Where it surfaces |
|---|---|
| `POST /diagnoses` | `/diagnose` — drag-drop or camera capture, with client-side validation mirroring the server's rules |
| `GET /diagnoses/{id}/stream` | `hooks/use-diagnosis-live.ts` — SSE via `fetch` (EventSource can't send a bearer token), falling back to polling |
| `GET /diagnoses/{id}` | `/diagnoses/[id]`, and the poll fallback above |
| `GET /diagnoses/{id}/image` | `components/diagnosis-image.tsx` — fetched as a blob, since `<img src>` can't authenticate |
| `GET /diagnoses` | `/diagnoses` — filters on status, class, plot, farm and date range |
| `DELETE /diagnoses/{id}` | `/diagnoses/[id]` |
| `POST /diagnoses/{id}/feedback` | `components/feedback-form.tsx` |
| `GET /diseases`, `/diseases/{id}` | `/diseases` |
| `GET /auth/me`, `POST /auth/register` | `/account`, `/register` |
| `/farms/*` | `/farms` — farm and plot management, with plots loaded lazily per farm |
| `GET /admin/models`, `POST /admin/models/{v}/activate` | `/admin` (admin claim only) |
| `GET /health` | Sidebar status popover |
| `/admin/stats` | `/admin` — wired to the real call, which answers 501, and the UI says so |

The remaining 501 route is deliberately **not** mocked. An empty stats panel
would read as "there is no activity", which is a different and false
statement; the page renders the backend's own 501 explanation instead.

## Design decisions worth knowing

**Uncertainty is the product.** Field accuracy is 65.3% against a 0.95
confidence threshold, so a large share of real photos come back `uncertain`
with the verdict withheld. `components/threshold-gauge.tsx` draws confidence
against that threshold as a gate the prediction either clears or doesn't, and
`uncertain` gets a designed state of its own — not an error style. Treating it
as a failure would teach operators to read the system as broken when it is
being careful.

**Colour is reserved.** Green, amber and rust mean healthy, uncertain and
diseased, and appear nowhere else. The brand accent is LED violet — the red +
blue spectrum a grow rack actually runs — so it can never be confused with a
verdict.

**Raw labels stay raw.** `Tomato___Late_blight` is shown in mono, with the
separator dimmed but not removed. It's the exact string the feedback endpoint
validates against, so tidying it would make corrections harder to get right.

**Hedges are surfaced, not hidden.** Classes with `fieldValidated: false` (10
of 38 have no field training data) carry a visible notice, and disease KB
entries say plainly that their agronomic content hasn't been written or
reviewed.

## Scripts

```bash
npm run dev        # dev server
npm run build      # production build
npm run lint       # eslint
npm run typecheck  # tsc --noEmit
```
