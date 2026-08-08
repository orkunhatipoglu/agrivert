"use client"

import * as React from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
  onIdTokenChanged,
  signInWithEmailAndPassword,
  signOut,
  type User,
} from "firebase/auth"

import { authApi } from "@/lib/api"
import { getFirebaseAuth, isFirebaseConfigured } from "@/lib/firebase"
import type { UserProfile } from "@/lib/types"

interface AuthContextValue {
  /** The Firebase user, or null when signed out. */
  user: User | null
  /** `GET /auth/me` — carries the `isAdmin` custom claim. */
  profile: UserProfile | null
  loading: boolean
  configured: boolean
  signIn: (email: string, password: string) => Promise<void>
  register: (
    email: string,
    password: string,
    displayName?: string
  ) => Promise<void>
  signOutUser: () => Promise<void>
  refreshProfile: () => Promise<void>
}

const AuthContext = React.createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const queryClient = useQueryClient()
  const [user, setUser] = React.useState<User | null>(null)
  // Without config there is nothing to wait for, so this never starts loading.
  const [loading, setLoading] = React.useState(isFirebaseConfigured)

  React.useEffect(() => {
    if (!isFirebaseConfigured) return
    // onIdTokenChanged rather than onAuthStateChanged: it also fires on token
    // refresh, so a claim change (e.g. being granted admin) is picked up
    // without a full sign-out. setState here runs from the subscription
    // callback, not synchronously during the effect.
    return onIdTokenChanged(getFirebaseAuth(), (next) => {
      setUser(next)
      setLoading(false)
    })
  }, [])

  // Scoped to the uid so switching accounts can't show a stale profile.
  const profileQuery = useQuery({
    queryKey: ["me", user?.uid],
    queryFn: authApi.me,
    enabled: Boolean(user),
    // A failed /auth/me must not lock the user out of the shell — the API may
    // simply be down. Pages that need `isAdmin` handle a null profile.
    retry: false,
    staleTime: 5 * 60_000,
  })

  const value = React.useMemo<AuthContextValue>(
    () => ({
      user,
      profile: user ? (profileQuery.data ?? null) : null,
      loading,
      configured: isFirebaseConfigured,
      signIn: async (email, password) => {
        await signInWithEmailAndPassword(getFirebaseAuth(), email, password)
      },
      register: async (email, password, displayName) => {
        // Registration goes through the API so the backend owns user creation,
        // then we sign in client-side to obtain the ID token.
        await authApi.register({ email, password, displayName })
        await signInWithEmailAndPassword(getFirebaseAuth(), email, password)
      },
      signOutUser: async () => {
        // POST /auth/logout is a 501 stub, so this is a client-side sign-out.
        // The API's `check_revoked=True` still rejects revoked tokens.
        await signOut(getFirebaseAuth())
        // Drop every cached response — it all belongs to the previous user.
        queryClient.clear()
      },
      refreshProfile: async () => {
        await profileQuery.refetch()
      },
    }),
    [user, profileQuery, loading, queryClient]
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const context = React.useContext(AuthContext)
  if (!context) {
    throw new Error("useAuth must be used inside <AuthProvider>")
  }
  return context
}

/** Turns Firebase's error codes into something worth reading. */
export function authErrorMessage(error: unknown): string {
  const code =
    error && typeof error === "object" && "code" in error
      ? String((error as { code: unknown }).code)
      : ""

  switch (code) {
    case "auth/invalid-credential":
    case "auth/wrong-password":
    case "auth/user-not-found":
      return "That email and password don't match an account."
    case "auth/invalid-email":
      return "Enter a valid email address."
    case "auth/user-disabled":
      return "This account has been disabled."
    case "auth/too-many-requests":
      return "Too many attempts. Wait a minute and try again."
    case "auth/network-request-failed":
      return "Can't reach Firebase. Check your connection."
    case "auth/email-already-in-use":
      return "An account with that email already exists."
    case "auth/weak-password":
      return "Use a password of at least 8 characters."
    default:
      return error instanceof Error ? error.message : "Something went wrong."
  }
}
