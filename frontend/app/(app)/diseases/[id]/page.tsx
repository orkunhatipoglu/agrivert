"use client"

import Link from "next/link"
import { useParams } from "next/navigation"
import { useQuery } from "@tanstack/react-query"
import { ArrowLeftIcon, PencilLineIcon, TriangleAlertIcon } from "lucide-react"

import { PageHeader } from "@/components/page-header"
import { RawLabel } from "@/components/raw-label"
import { FieldValidationNotice } from "@/components/verdict-panel"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { diseasesApi } from "@/lib/api"
import type { Severity } from "@/lib/types"

const SEVERITY_LABEL: Record<Severity, string> = {
  unknown: "Not assessed",
  low: "Low",
  moderate: "Moderate",
  high: "High",
}

export default function DiseaseDetailPage() {
  const params = useParams<{ id: string }>()
  const id = decodeURIComponent(params.id)

  const { data, isLoading, error } = useQuery({
    queryKey: ["disease", id],
    queryFn: () => diseasesApi.get(id),
  })

  if (isLoading) {
    return (
      <div className="flex flex-col gap-6">
        <Skeleton className="h-8 w-28" />
        <Skeleton className="h-12 w-80" />
        <Skeleton className="h-40 w-full" />
      </div>
    )
  }

  if (error || !data) {
    return (
      <>
        <BackLink />
        <Alert variant="destructive">
          <TriangleAlertIcon />
          <AlertTitle>Couldn&apos;t load this class</AlertTitle>
          <AlertDescription>
            {(error as Error | null)?.message ??
              "It isn't in the knowledge base."}
          </AlertDescription>
        </Alert>
      </>
    )
  }

  return (
    <>
      <BackLink />
      <PageHeader
        eyebrow={data.crop}
        title={data.condition}
        description={<RawLabel rawLabel={data.rawLabel} />}
      />

      <div className="flex flex-wrap gap-2">
        {data.healthy && (
          <Badge
            variant="outline"
            className="border-healthy/35 text-healthy bg-healthy/10"
          >
            Healthy class
          </Badge>
        )}
        <Badge variant="outline">
          Severity: {SEVERITY_LABEL[data.severity] ?? data.severity}
        </Badge>
      </div>

      {!data.fieldValidated && <FieldValidationNotice />}

      {!data.contentReviewed && (
        <Alert>
          <PencilLineIcon />
          <AlertTitle>Not written yet</AlertTitle>
          <AlertDescription>
            <p>
              This entry has its structure seeded from the model&apos;s labels,
              but no agronomist has written or reviewed its content. Nothing
              below is treatment advice.
            </p>
            <p>
              Fill in the Firestore document for{" "}
              <code className="font-mono text-xs">{data.diseaseId}</code> and
              set <code className="font-mono text-xs">content_reviewed</code> to
              true.
            </p>
          </AlertDescription>
        </Alert>
      )}

      {data.description && (
        <section className="flex max-w-prose flex-col gap-3">
          <h2 className="font-display text-lg font-semibold">About</h2>
          <p className="text-sm leading-relaxed">{data.description}</p>
        </section>
      )}

      <div className="grid gap-8 lg:grid-cols-3">
        <ListSection title="Symptoms" items={data.symptoms} />
        <ListSection title="Treatment" items={data.treatment} />
        <ListSection title="Prevention" items={data.prevention} />
      </div>

      {data.references.length > 0 && (
        <section className="flex flex-col gap-3">
          <h2 className="font-display text-lg font-semibold">References</h2>
          <ul className="flex flex-col gap-2">
            {data.references.map((reference) => (
              <li key={reference} className="text-sm break-all">
                {reference}
              </li>
            ))}
          </ul>
        </section>
      )}

      <div>
        <Button variant="outline" size="sm" asChild>
          <Link
            href={`/diagnoses?diseaseId=${encodeURIComponent(data.rawLabel)}`}
          >
            See your diagnoses of this class
          </Link>
        </Button>
      </div>
    </>
  )
}

function ListSection({ title, items }: { title: string; items: string[] }) {
  return (
    <section className="flex flex-col gap-3">
      <h2 className="font-display text-lg font-semibold">{title}</h2>
      {items.length ? (
        <ul className="flex list-disc flex-col gap-2 pl-4 text-sm">
          {items.map((item) => (
            <li key={item} className="leading-relaxed">
              {item}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-muted-foreground text-sm">Nothing recorded.</p>
      )}
    </section>
  )
}

function BackLink() {
  return (
    <Button variant="ghost" size="sm" asChild className="-ml-2 self-start">
      <Link href="/diseases">
        <ArrowLeftIcon data-icon="inline-start" />
        Disease library
      </Link>
    </Button>
  )
}
