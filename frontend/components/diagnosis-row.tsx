"use client"

import Link from "next/link"
import { formatDistanceToNow } from "date-fns"
import { ChevronRightIcon } from "lucide-react"

import { DiagnosisImage } from "@/components/diagnosis-image"
import { StatusBadge, VerdictBadge } from "@/components/status-badge"
import { formatConfidence } from "@/lib/labels"
import type { DiagnosisListItem } from "@/lib/types"

/**
 * One row of history.
 *
 * The headline is the verdict where there is one, and an explicit statement
 * of its absence where there isn't — a row that just said "Uncertain" in grey
 * would be read as "still loading".
 */
export function DiagnosisRow({
  item,
  showThumbnail = false,
}: {
  item: DiagnosisListItem
  /** Costs one authenticated fetch per row, so only for short lists. */
  showThumbnail?: boolean
}) {
  const headline = describe(item)

  return (
    <Link
      href={`/diagnoses/${item.diagnosisId}`}
      className="hover:bg-muted/50 focus-visible:ring-ring group flex items-center gap-4 rounded-lg p-3 transition-colors focus-visible:ring-2 focus-visible:outline-none"
    >
      {showThumbnail && (
        <DiagnosisImage
          diagnosisId={item.diagnosisId}
          alt=""
          className="size-12 shrink-0"
        />
      )}

      <div className="flex min-w-0 flex-1 flex-col gap-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="truncate text-sm font-medium">{headline}</span>
          <VerdictBadge healthy={item.healthy} />
        </div>
        <div className="text-muted-foreground flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-xs">
          <time dateTime={item.createdAt}>
            {formatDistanceToNow(new Date(item.createdAt), { addSuffix: true })}
          </time>
          {typeof item.confidence === "number" && (
            <span className="tabular-nums">
              {formatConfidence(item.confidence)}
            </span>
          )}
        </div>
      </div>

      <StatusBadge status={item.status} className="hidden shrink-0 sm:flex" />
      <ChevronRightIcon className="text-muted-foreground size-4 shrink-0" />
    </Link>
  )
}

function describe(item: DiagnosisListItem): string {
  switch (item.status) {
    case "completed":
      if (item.healthy) return `${item.crop ?? "Plant"} — healthy`
      return item.condition ?? "Condition detected"
    case "uncertain":
      return "No confident verdict"
    case "rejected":
      return "Photo rejected"
    case "failed":
      return "Analysis failed"
    case "processing":
      return "Analysing"
    default:
      return "Queued"
  }
}
