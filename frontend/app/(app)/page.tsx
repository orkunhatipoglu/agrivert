"use client"

import Link from "next/link"
import { useQuery } from "@tanstack/react-query"
import { ArrowRightIcon, ScanIcon, TriangleAlertIcon } from "lucide-react"

import { DiagnosisRow } from "@/components/diagnosis-row"
import { PageHeader } from "@/components/page-header"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { diagnosesApi } from "@/lib/api"
import { useAuth } from "@/hooks/use-auth"
import type { DiagnosisList } from "@/lib/types"
import { cn } from "@/lib/utils"

export default function OverviewPage() {
  const { profile, user } = useAuth()
  const name = (profile?.displayName ?? user?.email ?? "").split("@")[0]

  const { data, isLoading, error } = useQuery({
    queryKey: ["diagnoses", { limit: 200 }],
    queryFn: () => diagnosesApi.list({ limit: 200 }),
  })

  const stats = summarise(data)

  return (
    <>
      <PageHeader
        eyebrow="Overview"
        title={name ? `Welcome back, ${name}` : "Overview"}
        description="Everything you've photographed, and what the model made of it."
        actions={
          <Button asChild>
            <Link href="/diagnose">
              <ScanIcon data-icon="inline-start" />
              Diagnose a plant
            </Link>
          </Button>
        }
      />

      {error && (
        <Alert variant="destructive">
          <TriangleAlertIcon />
          <AlertTitle>Couldn&apos;t load your history</AlertTitle>
          <AlertDescription>{(error as Error).message}</AlertDescription>
        </Alert>
      )}

      <section className="grid gap-px overflow-hidden rounded-xl border sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          label="Diagnoses"
          value={stats.total}
          detail="all time"
          loading={isLoading}
        />
        <Stat
          label="Needs attention"
          value={stats.diseased}
          detail="disease reported"
          tone="diseased"
          loading={isLoading}
        />
        <Stat
          label="Healthy"
          value={stats.healthy}
          detail="cleared"
          tone="healthy"
          loading={isLoading}
        />
        <Stat
          label="Withheld"
          value={stats.uncertain}
          detail="below threshold"
          tone="uncertain"
          loading={isLoading}
        />
      </section>

      {stats.total > 0 && stats.uncertain / stats.total > 0.4 && (
        <Alert>
          <TriangleAlertIcon />
          <AlertTitle>
            Most of your photos are coming back uncertain
          </AlertTitle>
          <AlertDescription>
            <p>
              {stats.uncertain} of {stats.total} landed below the confidence
              threshold. That is expected on real field photographs, but tighter
              framing and flatter light shift a good share of them over the line.
            </p>
          </AlertDescription>
        </Alert>
      )}

      <section className="flex flex-col gap-4">
        <div className="flex items-end justify-between gap-4">
          <h2 className="font-display text-xl font-semibold">Recent activity</h2>
          {stats.total > 0 && (
            <Button variant="ghost" size="sm" asChild>
              <Link href="/diagnoses">
                All history
                <ArrowRightIcon data-icon="inline-end" />
              </Link>
            </Button>
          )}
        </div>

        {isLoading ? (
          <div className="flex flex-col gap-2">
            {Array.from({ length: 4 }).map((_, index) => (
              <Skeleton key={index} className="h-[4.5rem] rounded-lg" />
            ))}
          </div>
        ) : !data?.items.length ? (
          <Empty className="rounded-xl border border-dashed">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <ScanIcon />
              </EmptyMedia>
              <EmptyTitle>No diagnoses yet</EmptyTitle>
              <EmptyDescription>
                Photograph a leaf on any rack layer to get your first verdict.
              </EmptyDescription>
            </EmptyHeader>
            <EmptyContent>
              <Button asChild>
                <Link href="/diagnose">Diagnose a plant</Link>
              </Button>
            </EmptyContent>
          </Empty>
        ) : (
          <div className="rounded-xl border p-1.5">
            {data.items.slice(0, 6).map((item, index) => (
              <div key={item.diagnosisId}>
                {index > 0 && <Separator />}
                <DiagnosisRow item={item} showThumbnail />
              </div>
            ))}
          </div>
        )}
      </section>
    </>
  )
}

function Stat({
  label,
  value,
  detail,
  tone = "neutral",
  loading,
}: {
  label: string
  value: number
  detail: string
  tone?: "neutral" | "healthy" | "diseased" | "uncertain"
  loading?: boolean
}) {
  const toneClass = {
    neutral: "text-foreground",
    healthy: "text-healthy",
    diseased: "text-diseased",
    uncertain: "text-uncertain",
  }[tone]

  return (
    <div className="bg-card flex flex-col gap-3 p-5">
      <span className="label-micro">{label}</span>
      {loading ? (
        <Skeleton className="h-9 w-14" />
      ) : (
        <span
          className={cn(
            "font-mono text-3xl leading-none font-medium tabular-nums",
            toneClass
          )}
        >
          {value}
        </span>
      )}
      <span className="text-muted-foreground text-xs">{detail}</span>
    </div>
  )
}

function summarise(data: DiagnosisList | undefined) {
  const items = data?.items ?? []
  return {
    total: items.length,
    healthy: items.filter((i) => i.status === "completed" && i.healthy === true)
      .length,
    diseased: items.filter(
      (i) => i.status === "completed" && i.healthy === false
    ).length,
    uncertain: items.filter((i) => i.status === "uncertain").length,
  }
}
