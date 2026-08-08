"use client"

import { useRouter } from "next/navigation"
import { useQuery } from "@tanstack/react-query"
import { LogOutIcon, ShieldIcon, TriangleAlertIcon } from "lucide-react"
import { toast } from "sonner"

import { PageHeader } from "@/components/page-header"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { useAuth } from "@/hooks/use-auth"
import { API_BASE, API_PREFIX, authApi } from "@/lib/api"

export default function AccountPage() {
  const { signOutUser } = useAuth()
  const router = useRouter()

  const { data, isLoading, error } = useQuery({
    queryKey: ["me"],
    queryFn: authApi.me,
  })

  async function handleSignOut() {
    try {
      await signOutUser()
      router.push("/login")
    } catch {
      toast.error("Couldn't sign out. Try again.")
    }
  }

  return (
    <>
      <PageHeader
        eyebrow="Account"
        title="Your account"
        description="Read from GET /auth/me, so this reflects what the API sees — including your custom claims."
        actions={
          <Button variant="outline" onClick={handleSignOut}>
            <LogOutIcon data-icon="inline-start" />
            Sign out
          </Button>
        }
      />

      {error && (
        <Alert variant="destructive">
          <TriangleAlertIcon />
          <AlertTitle>Couldn&apos;t load your profile</AlertTitle>
          <AlertDescription>{(error as Error).message}</AlertDescription>
        </Alert>
      )}

      {isLoading ? (
        <Skeleton className="h-56 rounded-xl" />
      ) : (
        data && (
          <section className="rounded-xl border">
            <Row label="Name" value={data.displayName ?? "Not set"} />
            <Separator />
            <Row label="Email" value={data.email ?? "Not set"} />
            <Separator />
            <Row
              label="Email verified"
              value={data.emailVerified ? "Yes" : "No"}
            />
            <Separator />
            <Row label="UID" value={data.uid} mono />
            <Separator />
            <div className="flex flex-wrap items-center justify-between gap-4 p-4">
              <span className="label-micro">Role</span>
              {data.isAdmin ? (
                <Badge variant="outline" className="gap-1.5">
                  <ShieldIcon />
                  Admin
                </Badge>
              ) : (
                <span className="text-muted-foreground font-mono text-xs">
                  Operator
                </span>
              )}
            </div>
          </section>
        )
      )}

      <section className="flex flex-col gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-display text-lg font-semibold">
            How sign-in works here
          </h2>
          <p className="text-muted-foreground max-w-prose text-sm leading-relaxed">
            The API never issues tokens. The Firebase client SDK signs you in
            and refreshes your ID token; every request carries it as a bearer
            token, and the API verifies it — checking for revocation on each
            call. That is why{" "}
            <code className="font-mono text-xs">POST /auth/login</code>,{" "}
            <code className="font-mono text-xs">/auth/refresh</code> and{" "}
            <code className="font-mono text-xs">/auth/logout</code> answer 501:
            routing credentials through the backend would mean handling raw
            passwords for no benefit.
          </p>
        </div>

        <div className="text-muted-foreground flex flex-col gap-1.5 rounded-lg border p-4 font-mono text-xs">
          <span className="break-all">
            API {API_BASE}
            {API_PREFIX}
          </span>
        </div>
      </section>
    </>
  )
}

function Row({
  label,
  value,
  mono,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-4 p-4">
      <span className="label-micro">{label}</span>
      <span
        className={
          mono
            ? "text-muted-foreground font-mono text-xs break-all"
            : "text-sm break-words"
        }
      >
        {value}
      </span>
    </div>
  )
}
