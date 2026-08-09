"use client"

import * as React from "react"
import Link from "next/link"
import { useSearchParams } from "next/navigation"
import { useQuery } from "@tanstack/react-query"
import { HistoryIcon, TriangleAlertIcon, XIcon } from "lucide-react"

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
import { Field, FieldLabel } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { diagnosesApi, diseasesApi } from "@/lib/api"
import type { DiagnosisFilters, DiagnosisStatus } from "@/lib/types"

const ANY = "__any__"

const STATUSES: { value: DiagnosisStatus; label: string }[] = [
  { value: "completed", label: "Complete" },
  { value: "uncertain", label: "Uncertain" },
  { value: "processing", label: "Analysing" },
  { value: "queued", label: "Queued" },
  { value: "rejected", label: "Rejected" },
  { value: "failed", label: "Failed" },
]

export default function HistoryPage() {
  return (
    <React.Suspense fallback={null}>
      <History />
    </React.Suspense>
  )
}

function History() {
  // Deep link from the disease library: /diagnoses?diseaseId=Tomato___Late_blight
  const searchParams = useSearchParams()
  const [status, setStatus] = React.useState<string>(ANY)
  const [diseaseId, setDiseaseId] = React.useState<string>(
    () => searchParams.get("diseaseId") ?? ANY
  )
  const [dateFrom, setDateFrom] = React.useState("")
  const [dateTo, setDateTo] = React.useState("")

  const filters: DiagnosisFilters = {
    status: status === ANY ? undefined : (status as DiagnosisStatus),
    diseaseId: diseaseId === ANY ? undefined : diseaseId,
    // <input type="date"> gives a bare date; the API parses a datetime.
    dateFrom: dateFrom ? new Date(dateFrom).toISOString() : undefined,
    dateTo: dateTo ? new Date(`${dateTo}T23:59:59`).toISOString() : undefined,
    limit: 200,
  }

  const active =
    status !== ANY ||
    diseaseId !== ANY ||
    Boolean(dateFrom || dateTo)

  const { data, isLoading, error } = useQuery({
    queryKey: ["diagnoses", filters],
    queryFn: () => diagnosesApi.list(filters),
  })

  const { data: diseases } = useQuery({
    queryKey: ["diseases"],
    queryFn: diseasesApi.list,
    staleTime: 10 * 60_000,
  })

  function clearFilters() {
    setStatus(ANY)
    setDiseaseId(ANY)
    setDateFrom("")
    setDateTo("")
  }

  return (
    <>
      <PageHeader
        eyebrow="History"
        title="Diagnosis history"
        description="Every photo you've submitted, newest first."
        actions={
          active && (
            <Button variant="outline" onClick={clearFilters}>
              <XIcon data-icon="inline-start" />
              Clear filters
            </Button>
          )
        }
      />

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        <Field>
          <FieldLabel htmlFor="status">Status</FieldLabel>
          <Select value={status} onValueChange={setStatus}>
            <SelectTrigger id="status">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ANY}>Any status</SelectItem>
              {STATUSES.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>

        <Field>
          <FieldLabel htmlFor="disease">Class</FieldLabel>
          <Select value={diseaseId} onValueChange={setDiseaseId}>
            <SelectTrigger id="disease">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ANY}>Any class</SelectItem>
              {diseases?.items.map((disease) => (
                <SelectItem key={disease.rawLabel} value={disease.rawLabel}>
                  {disease.crop} — {disease.condition}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>

        <Field>
          <FieldLabel htmlFor="from">From</FieldLabel>
          <Input
            id="from"
            type="date"
            value={dateFrom}
            onChange={(event) => setDateFrom(event.target.value)}
          />
        </Field>

        <Field>
          <FieldLabel htmlFor="to">To</FieldLabel>
          <Input
            id="to"
            type="date"
            value={dateTo}
            onChange={(event) => setDateTo(event.target.value)}
          />
        </Field>
      </section>

      {error && (
        <Alert variant="destructive">
          <TriangleAlertIcon />
          <AlertTitle>Couldn&apos;t load history</AlertTitle>
          <AlertDescription>
            <p>{(error as Error).message}</p>
            <p>
              Combining filters needs a matching Firestore composite index. The
              API log prints a link that creates the exact one required.
            </p>
          </AlertDescription>
        </Alert>
      )}

      {isLoading ? (
        <div className="flex flex-col gap-2">
          {Array.from({ length: 6 }).map((_, index) => (
            <Skeleton key={index} className="h-16 rounded-lg" />
          ))}
        </div>
      ) : !data?.items.length ? (
        <Empty className="rounded-xl border border-dashed">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <HistoryIcon />
            </EmptyMedia>
            <EmptyTitle>
              {active ? "Nothing matches those filters" : "No diagnoses yet"}
            </EmptyTitle>
            <EmptyDescription>
              {active
                ? "Widen the range or clear the filters to see everything."
                : "Your first verdict will show up here."}
            </EmptyDescription>
          </EmptyHeader>
          <EmptyContent>
            {active ? (
              <Button variant="outline" onClick={clearFilters}>
                Clear filters
              </Button>
            ) : (
              <Button asChild>
                <Link href="/diagnose">Diagnose a plant</Link>
              </Button>
            )}
          </EmptyContent>
        </Empty>
      ) : (
        <section className="flex flex-col gap-3">
          <p className="text-muted-foreground font-mono text-xs">
            {data.items.length} result{data.items.length === 1 ? "" : "s"}
          </p>
          <div className="rounded-xl border p-1.5">
            {data.items.map((item, index) => (
              <div key={item.diagnosisId}>
                {index > 0 && <Separator />}
                <DiagnosisRow item={item} />
              </div>
            ))}
          </div>
        </section>
      )}
    </>
  )
}
