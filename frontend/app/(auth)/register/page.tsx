"use client"

import * as React from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { TriangleAlertIcon } from "lucide-react"

import { FirebaseSetupNotice } from "@/components/auth-guard"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Spinner } from "@/components/ui/spinner"
import { ApiError } from "@/lib/api"
import { authErrorMessage, useAuth } from "@/hooks/use-auth"

export default function RegisterPage() {
  const { register, user, configured, loading } = useAuth()
  const router = useRouter()

  const [email, setEmail] = React.useState("")
  const [displayName, setDisplayName] = React.useState("")
  const [password, setPassword] = React.useState("")
  const [error, setError] = React.useState<string | null>(null)
  const [submitting, setSubmitting] = React.useState(false)

  React.useEffect(() => {
    if (!loading && user) router.replace("/")
  }, [loading, user, router])

  if (!configured) return <FirebaseSetupNotice />

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    setError(null)

    // Mirrors the backend's `Field(min_length=8)` so the check happens before
    // a round trip, with the same rule.
    if (password.length < 8) {
      setError("Use a password of at least 8 characters.")
      return
    }

    setSubmitting(true)
    try {
      await register(email, password, displayName || undefined)
      router.replace("/")
    } catch (caught) {
      setError(
        caught instanceof ApiError ? caught.message : authErrorMessage(caught)
      )
      setSubmitting(false)
    }
  }

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col gap-2">
        <h1 className="font-display text-2xl font-semibold">Create account</h1>
        <p className="text-muted-foreground text-sm">
          One account per operator. Diagnoses are private to it.
        </p>
      </div>

      {error && (
        <Alert variant="destructive">
          <TriangleAlertIcon />
          <AlertTitle>Couldn&apos;t create the account</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <form onSubmit={handleSubmit}>
        <FieldGroup>
          <Field>
            <FieldLabel htmlFor="displayName">Name</FieldLabel>
            <Input
              id="displayName"
              autoComplete="name"
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
            />
            <FieldDescription>Optional.</FieldDescription>
          </Field>
          <Field>
            <FieldLabel htmlFor="email">Email</FieldLabel>
            <Input
              id="email"
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(event) => setEmail(event.target.value)}
            />
          </Field>
          <Field>
            <FieldLabel htmlFor="password">Password</FieldLabel>
            <Input
              id="password"
              type="password"
              autoComplete="new-password"
              required
              minLength={8}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
            <FieldDescription>At least 8 characters.</FieldDescription>
          </Field>
          <Field>
            <Button type="submit" disabled={submitting}>
              {submitting && <Spinner />}
              Create account
            </Button>
          </Field>
        </FieldGroup>
      </form>

      <p className="text-muted-foreground text-sm">
        Already have one?{" "}
        <Link
          href="/login"
          className="text-foreground underline underline-offset-4"
        >
          Sign in
        </Link>
      </p>
    </div>
  )
}
