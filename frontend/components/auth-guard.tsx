"use client"

import * as React from "react"
import { usePathname, useRouter } from "next/navigation"
import { TriangleAlertIcon } from "lucide-react"

import { BrandMark } from "@/components/brand"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { useAuth } from "@/hooks/use-auth"

/**
 * Client-side gate. Everything behind it calls an API that verifies the
 * Firebase ID token itself, so this is a redirect for the user's benefit, not
 * the security boundary — the boundary is `get_current_user` on the server.
 */
export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { user, loading, configured } = useAuth()
  const router = useRouter()
  const pathname = usePathname()

  React.useEffect(() => {
    if (!configured || loading || user) return
    const next = encodeURIComponent(pathname)
    router.replace(`/login?next=${next}`)
  }, [configured, loading, user, pathname, router])

  if (!configured) {
    return <FirebaseSetupNotice />
  }

  if (loading || !user) {
    return (
      <div className="flex min-h-svh items-center justify-center">
        <BrandMark className="text-muted-foreground size-8 animate-pulse" />
        <span className="sr-only">Checking your session</span>
      </div>
    )
  }

  return <>{children}</>
}

/**
 * Without Firebase config the app cannot obtain a token, so every API call
 * would 401. Say that once, with the fix, instead of failing on each page.
 */
export function FirebaseSetupNotice() {
  return (
    <div className="mx-auto flex min-h-svh max-w-lg items-center px-6">
      <Alert variant="destructive">
        <TriangleAlertIcon />
        <AlertTitle>Firebase isn&apos;t configured</AlertTitle>
        <AlertDescription>
          <p>
            The API verifies a Firebase ID token on every request, and the
            browser can&apos;t mint one without client config.
          </p>
          <p>
            Copy <code className="font-mono">.env.example</code> to{" "}
            <code className="font-mono">.env.local</code>, fill in the{" "}
            <code className="font-mono">NEXT_PUBLIC_FIREBASE_*</code> values
            from your Firebase project settings, then restart the dev server.
          </p>
        </AlertDescription>
      </Alert>
    </div>
  )
}
