"use client"

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { format } from "date-fns"
import { CheckIcon, CpuIcon, ShieldIcon, TriangleAlertIcon } from "lucide-react"
import { toast } from "sonner"

import { NotImplemented, isNotImplemented } from "@/components/not-implemented"
import { PageHeader } from "@/components/page-header"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { Spinner } from "@/components/ui/spinner"
import { useAuth } from "@/hooks/use-auth"
import { adminApi } from "@/lib/api"
import { formatMetricKey, formatMetricValue } from "@/lib/labels"
import type { ModelVersionInfo } from "@/lib/types"

export default function AdminPage() {
  const { profile } = useAuth()
  const queryClient = useQueryClient()

  const { data, isLoading, error } = useQuery({
    queryKey: ["admin", "models"],
    queryFn: adminApi.models,
  })

  const stats = useQuery({
    queryKey: ["admin", "stats"],
    queryFn: adminApi.stats,
  })

  const activate = useMutation({
    mutationFn: (version: string) => adminApi.activate(version),
    onSuccess: (response) => {
      toast.success(`${response.version} is now active.`, {
        description: response.detail,
        duration: 8000,
      })
      void queryClient.invalidateQueries({ queryKey: ["admin", "models"] })
      void queryClient.invalidateQueries({ queryKey: ["health"] })
    },
    onError: (caught: Error) => toast.error(caught.message),
  })

  if (profile && !profile.isAdmin) {
    return (
      <>
        <PageHeader eyebrow="Admin" title="Model registry" />
        <Empty className="rounded-xl border border-dashed">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <ShieldIcon />
            </EmptyMedia>
            <EmptyTitle>Admin only</EmptyTitle>
            <EmptyDescription>
              These routes require the <code className="font-mono">admin</code>{" "}
              custom claim on your Firebase account.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      </>
    )
  }

  return (
    <>
      <PageHeader
        eyebrow="Admin"
        title="Model registry"
        description="Every version on disk, and which one is serving. Swapping a retrained model is a file drop plus an activation."
      />

      {error && !isNotImplemented(error) && (
        <Alert variant="destructive">
          <TriangleAlertIcon />
          <AlertTitle>Couldn&apos;t load model versions</AlertTitle>
          <AlertDescription>{(error as Error).message}</AlertDescription>
        </Alert>
      )}

      {isLoading ? (
        <div className="flex flex-col gap-3">
          {Array.from({ length: 2 }).map((_, index) => (
            <Skeleton key={index} className="h-44 rounded-xl" />
          ))}
        </div>
      ) : !data?.items.length ? (
        <Empty className="rounded-xl border border-dashed">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <CpuIcon />
            </EmptyMedia>
            <EmptyTitle>No versions registered</EmptyTitle>
            <EmptyDescription>
              Register one with{" "}
              <code className="font-mono text-xs">
                python scripts/register_model.py ../artifacts --version
                v1-... --activate
              </code>
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : (
        <section className="flex flex-col gap-4">
          {data.items.map((version) => (
            <ModelCard
              key={version.version}
              version={version}
              onActivate={() => activate.mutate(version.version)}
              activating={
                activate.isPending && activate.variables === version.version
              }
            />
          ))}
        </section>
      )}

      <section className="flex flex-col gap-4">
        <h2 className="font-display text-xl font-semibold">
          Accuracy & feedback
        </h2>
        {stats.isLoading ? (
          <Skeleton className="h-32 rounded-xl" />
        ) : isNotImplemented(stats.error) ? (
          <NotImplemented route="GET /api/v1/admin/stats" error={stats.error} />
        ) : stats.error ? (
          <Alert variant="destructive">
            <TriangleAlertIcon />
            <AlertTitle>Couldn&apos;t load stats</AlertTitle>
            <AlertDescription>{(stats.error as Error).message}</AlertDescription>
          </Alert>
        ) : (
          stats.data && (
            <div className="grid gap-px overflow-hidden rounded-xl border sm:grid-cols-2 lg:grid-cols-4">
              {[
                { label: "Diagnoses", value: stats.data.diagnoses.totalDiagnoses },
                { label: "Completed", value: stats.data.diagnoses.completed },
                { label: "Uncertain", value: stats.data.diagnoses.uncertain },
                { label: "Feedback", value: stats.data.diagnoses.feedbackCount },
              ].map((stat) => (
                <div key={stat.label} className="bg-card flex flex-col gap-3 p-5">
                  <span className="label-micro">{stat.label}</span>
                  <span className="font-mono text-2xl font-medium tabular-nums">
                    {stat.value}
                  </span>
                </div>
              ))}
            </div>
          )
        )}
      </section>
    </>
  )
}

function ModelCard({
  version,
  onActivate,
  activating,
}: {
  version: ModelVersionInfo
  onActivate: () => void
  activating: boolean
}) {
  const metrics = Object.entries(version.metrics ?? {})

  return (
    <article className="flex flex-col gap-5 rounded-xl border p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-mono text-base font-medium">
              {version.version}
            </h3>
            {version.active && (
              <Badge
                variant="outline"
                className="border-healthy/35 text-healthy bg-healthy/10 gap-1.5"
              >
                <CheckIcon />
                Serving
              </Badge>
            )}
          </div>
          <p className="text-muted-foreground font-mono text-xs">
            {[
              version.architecture,
              version.numClasses ? `${version.numClasses} classes` : null,
              version.bestEpoch ? `best epoch ${version.bestEpoch}` : null,
              version.registeredAt
                ? format(new Date(version.registeredAt), "d MMM yyyy")
                : null,
            ]
              .filter(Boolean)
              .join(" · ")}
          </p>
        </div>

        {!version.active && (
          <Button
            variant="outline"
            size="sm"
            disabled={activating}
            onClick={onActivate}
          >
            {activating && <Spinner />}
            Activate
          </Button>
        )}
      </div>

      {metrics.length > 0 && (
        <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {metrics.map(([key, value]) => (
            <div key={key} className="flex flex-col gap-1.5">
              <dt className="label-micro">{formatMetricKey(key)}</dt>
              <dd className="font-mono text-lg tabular-nums">
                {formatMetricValue(value)}
              </dd>
            </div>
          ))}
        </dl>
      )}

      <div className="text-muted-foreground flex flex-wrap gap-x-6 gap-y-2 font-mono text-xs">
        {version.confidenceThreshold !== null &&
          version.confidenceThreshold !== undefined && (
            <span>threshold {version.confidenceThreshold}</span>
          )}
        {version.temperature !== null && version.temperature !== undefined && (
          <span>temperature {version.temperature}</span>
        )}
      </div>

      {/* Straight from metadata.json — the studio/field gap is the single most
          important caveat on any of these numbers. */}
      {version.caveat && (
        <Alert>
          <TriangleAlertIcon />
          <AlertTitle>Caveat</AlertTitle>
          <AlertDescription>{version.caveat}</AlertDescription>
        </Alert>
      )}
    </article>
  )
}
