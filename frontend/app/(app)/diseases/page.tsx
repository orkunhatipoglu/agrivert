"use client"

import * as React from "react"
import Link from "next/link"
import { useQuery } from "@tanstack/react-query"
import {
  BookOpenIcon,
  ChevronRightIcon,
  FlaskConicalIcon,
  TriangleAlertIcon,
} from "lucide-react"

import { PageHeader } from "@/components/page-header"
import { RawLabel } from "@/components/raw-label"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { diseasesApi } from "@/lib/api"
import type { DiseaseSummary } from "@/lib/types"

export default function DiseasesPage() {
  const [search, setSearch] = React.useState("")

  const { data, isLoading, error } = useQuery({
    queryKey: ["diseases"],
    queryFn: diseasesApi.list,
    staleTime: 10 * 60_000,
  })

  const groups = React.useMemo(() => {
    const term = search.trim().toLowerCase()
    const items = (data?.items ?? []).filter(
      (item) =>
        !term ||
        item.crop.toLowerCase().includes(term) ||
        item.condition.toLowerCase().includes(term) ||
        item.rawLabel.toLowerCase().includes(term)
    )
    const byCrop = new Map<string, DiseaseSummary[]>()
    for (const item of items) {
      const list = byCrop.get(item.crop) ?? []
      list.push(item)
      byCrop.set(item.crop, list)
    }
    return [...byCrop.entries()].sort(([a], [b]) => a.localeCompare(b))
  }, [data, search])

  const studioOnly = (data?.items ?? []).filter(
    (item) => !item.fieldValidated
  ).length

  return (
    <>
      <PageHeader
        eyebrow="Reference"
        title="Disease library"
        description="Every class the active model can predict, grouped by crop."
      />

      {error && (
        <Alert variant="destructive">
          <TriangleAlertIcon />
          <AlertTitle>Couldn&apos;t load the library</AlertTitle>
          <AlertDescription>{(error as Error).message}</AlertDescription>
        </Alert>
      )}

      {/* The KB ships with the shape filled in and the agronomic text blank,
          on purpose. Saying so once at the top is more honest than letting
          every detail page look broken. */}
      {!isLoading && data?.items.length ? (
        <Alert>
          <BookOpenIcon />
          <AlertTitle>Treatment guidance hasn&apos;t been written yet</AlertTitle>
          <AlertDescription>
            <p>
              The library is seeded from the model&apos;s own label list, so it
              can never drift out of sync with what the model predicts. The
              description, symptoms and treatment fields are deliberately empty
              until an agronomist fills them in — text a farmer acts on in a
              real field isn&apos;t worth generating unreviewed.
            </p>
            {studioOnly > 0 && (
              <p>
                {studioOnly} of {data.items.length} classes are backed by studio
                images only.
              </p>
            )}
          </AlertDescription>
        </Alert>
      ) : null}

      <Input
        type="search"
        placeholder="Search crops and conditions"
        value={search}
        onChange={(event) => setSearch(event.target.value)}
        className="max-w-sm"
      />

      {isLoading ? (
        <div className="flex flex-col gap-3">
          {Array.from({ length: 5 }).map((_, index) => (
            <Skeleton key={index} className="h-20 rounded-lg" />
          ))}
        </div>
      ) : !groups.length ? (
        <Empty className="rounded-xl border border-dashed">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <BookOpenIcon />
            </EmptyMedia>
            <EmptyTitle>
              {search ? "No matches" : "The library is empty"}
            </EmptyTitle>
            <EmptyDescription>
              {search
                ? "Try a different crop or condition."
                : "Run scripts/seed_diseases.py on the backend to populate it from the active model's labels."}
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : (
        <div className="flex flex-col gap-10">
          {groups.map(([crop, items]) => (
            <section key={crop} className="flex flex-col gap-4">
              <div className="flex items-baseline gap-3">
                <h2 className="font-display text-xl font-semibold">{crop}</h2>
                <span className="text-muted-foreground font-mono text-xs">
                  {items.length} class{items.length === 1 ? "" : "es"}
                </span>
              </div>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {items.map((item) => (
                  <Link
                    key={item.diseaseId}
                    href={`/diseases/${encodeURIComponent(item.diseaseId)}`}
                    className="hover:border-primary/50 focus-visible:ring-ring group flex flex-col gap-3 rounded-lg border p-4 transition-colors focus-visible:ring-2 focus-visible:outline-none"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <span className="font-medium text-pretty">
                        {item.condition}
                      </span>
                      <ChevronRightIcon className="text-muted-foreground mt-0.5 size-4 shrink-0" />
                    </div>
                    <RawLabel rawLabel={item.rawLabel} className="self-start" />
                    <div className="flex flex-wrap gap-1.5">
                      {item.healthy && (
                        <Badge
                          variant="outline"
                          className="border-healthy/35 text-healthy bg-healthy/10"
                        >
                          Healthy class
                        </Badge>
                      )}
                      {!item.fieldValidated && (
                        <Badge variant="outline" className="gap-1.5">
                          <FlaskConicalIcon />
                          Studio only
                        </Badge>
                      )}
                    </div>
                  </Link>
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
    </>
  )
}
