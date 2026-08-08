/**
 * Firebase client SDK.
 *
 * The backend's `/auth/login`, `/auth/refresh` and `/auth/logout` return 501
 * on purpose: with Firebase Auth the *client* owns the credential lifecycle.
 * `signInWithEmailAndPassword` mints the ID token and `getIdToken()` refreshes
 * it; the API only ever verifies the bearer token it is handed.
 */

import { getApp, getApps, initializeApp, type FirebaseApp } from "firebase/app"
import { getAuth, type Auth } from "firebase/auth"

const config = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
}

/**
 * Whether the browser has enough config to talk to Firebase at all. Checked
 * before init so a missing `.env.local` produces one clear setup message
 * instead of an opaque SDK error on every page.
 */
export const isFirebaseConfigured = Boolean(
  config.apiKey && config.authDomain && config.projectId
)

let app: FirebaseApp | null = null
let auth: Auth | null = null

export function getFirebaseAuth(): Auth {
  if (!isFirebaseConfigured) {
    throw new Error(
      "Firebase is not configured. Copy .env.example to .env.local and fill in the NEXT_PUBLIC_FIREBASE_* values."
    )
  }
  if (!app) {
    app = getApps().length ? getApp() : initializeApp(config)
  }
  if (!auth) {
    auth = getAuth(app)
  }
  return auth
}

/**
 * A fresh ID token for the signed-in user, or null when signed out.
 * `getIdToken()` refreshes automatically once the token is within five
 * minutes of expiry, which is why nothing here caches it.
 */
export async function getIdToken(): Promise<string | null> {
  if (!isFirebaseConfigured) return null
  const current = getFirebaseAuth().currentUser
  if (!current) return null
  return current.getIdToken()
}
