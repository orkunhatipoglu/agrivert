"use client"

import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { BellIcon, TriangleAlertIcon } from "lucide-react"
import { toast } from "sonner"

import { NotImplemented, isNotImplemented } from "@/components/not-implemented"
import { PageHeader } from "@/components/page-header"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Field, FieldDescription, FieldLabel } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Spinner } from "@/components/ui/spinner"
import { notificationsApi } from "@/lib/api"

export default function NotificationsPage() {
  const [region, setRegion] = React.useState("")

  const { data, isLoading, error } = useQuery({
    queryKey: ["notifications"],
    queryFn: notificationsApi.list,
  })

  const subscribe = useMutation({
    mutationFn: () => notificationsApi.subscribe({ region: region || undefined }),
    onSuccess: () => toast.success("Subscribed to regional alerts."),
    onError: (caught: Error) => toast.error(caught.message),
  })

  return (
    <>
      <PageHeader
        eyebrow="Manage"
        title="Alerts"
        description="Outbreak warnings for your region, based on what other operators nearby are diagnosing."
      />

      {isLoading ? (
        <div className="flex flex-col gap-3">
          {Array.from({ length: 3 }).map((_, index) => (
            <Skeleton key={index} className="h-16 rounded-lg" />
          ))}
        </div>
      ) : isNotImplemented(error) ? (
        <>
          <NotImplemented route="GET /api/v1/notifications" error={error} />
          <Alert>
            <BellIcon />
            <AlertTitle>Why this one is last in line</AlertTitle>
            <AlertDescription>
              A regional outbreak alert needs two things that don&apos;t exist
              yet: farm locations, which depend on farm management shipping, and
              enough diagnosis volume per region for &ldquo;trending&rdquo; to
              mean anything. Building it earlier would produce confident alerts
              from a handful of photos.
            </AlertDescription>
          </Alert>
        </>
      ) : error ? (
        <Alert variant="destructive">
          <TriangleAlertIcon />
          <AlertTitle>Couldn&apos;t load alerts</AlertTitle>
          <AlertDescription>{(error as Error).message}</AlertDescription>
        </Alert>
      ) : (
        <div className="flex flex-col gap-3">
          {data?.items.map((notification) => (
            <article
              key={notification.notificationId}
              className="flex flex-col gap-1.5 rounded-lg border p-4"
            >
              <span className="label-micro">{notification.kind}</span>
              <h2 className="font-medium">{notification.title}</h2>
              {notification.body && (
                <p className="text-muted-foreground text-sm">
                  {notification.body}
                </p>
              )}
            </article>
          ))}
        </div>
      )}

      <section className="flex max-w-sm flex-col gap-4">
        <Field>
          <FieldLabel htmlFor="region">Region</FieldLabel>
          <Input
            id="region"
            value={region}
            onChange={(event) => setRegion(event.target.value)}
            placeholder="e.g. Antalya"
          />
          <FieldDescription>
            Posts to{" "}
            <code className="font-mono text-xs">
              POST /notifications/subscribe
            </code>
            .
          </FieldDescription>
        </Field>
        <Button
          variant="outline"
          className="self-start"
          disabled={subscribe.isPending}
          onClick={() => subscribe.mutate()}
        >
          {subscribe.isPending && <Spinner />}
          Subscribe to alerts
        </Button>
      </section>
    </>
  )
}
