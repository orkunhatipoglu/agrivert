"use client"

import Link from "next/link"
import {
  ArrowRightIcon,
  CircleHelpIcon,
  FlaskConicalIcon,
  ImageOffIcon,
  TriangleAlertIcon,
} from "lucide-react"

import { RawLabel } from "@/components/raw-label"
import { ThresholdGauge } from "@/components/threshold-gauge"
import { VerdictBadge } from "@/components/status-badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { formatConfidence, parseRawLabel } from "@/lib/labels"
import type { Diagnosis } from "@/lib/types"
import { cn } from "@/lib/utils"

/**
 * The verdict, in the four shapes it actually comes in.
 *
 * `uncertain` gets as much design attention as `completed` — with a 0.95
 * threshold against 65.3% field accuracy it is the common outcome on real
 * photographs, and treating it as a failure state would teach operators to
 * read the system as broken when it is being careful.
 */
export function VerdictPanel({ diagnosis }: { diagnosis: Diagnosis }) {
  switch (diagnosis.status) {
    case "completed":
      return <CompletedVerdict diagnosis={diagnosis} />
    case "uncertain":
      return <UncertainVerdict diagnosis={diagnosis} />
    case "rejected":
      return <RejectedVerdict diagnosis={diagnosis} />
    case "failed":
      return <FailedVerdict diagnosis={diagnosis} />
    default:
      return null
  }
}

function CompletedVerdict({ diagnosis }: { diagnosis: Diagnosis }) {
  const healthy = diagnosis.healthy === true
  const condition = diagnosis.condition ?? "Unknown condition"
  const crop = diagnosis.crop ?? "Unknown crop"

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <VerdictBadge healthy={diagnosis.healthy} />
          <span className="label-micro">{crop}</span>
        </div>
        <h2
          className={cn(
            "font-display text-4xl leading-tight font-semibold text-balance",
            healthy ? "text-healthy" : "text-diseased"
          )}
        >
          {healthy ? `${crop} looks healthy` : condition}
        </h2>
        {!healthy && (
          <p className="text-muted-foreground text-sm">
            Detected on {crop.toLowerCase()}.
          </p>
        )}
      </div>

      <ThresholdGauge
        confidence={diagnosis.confidence}
        threshold={diagnosis.threshold}
        tone={healthy ? "healthy" : "diseased"}
      />

      {diagnosis.fieldValidated === false && <FieldValidationNotice />}

      {diagnosis.recommendation ? (
        <div className="flex flex-col gap-2">
          <span className="label-micro">Recommended action</span>
          <p className="text-sm leading-relaxed">{diagnosis.recommendation}</p>
        </div>
      ) : (
        <p className="text-muted-foreground text-sm">
          No treatment guidance is attached to this verdict — the disease
          library hasn&apos;t been written yet.
        </p>
      )}

      <Alternatives diagnosis={diagnosis} title="Other candidates" />

      {diagnosis.diseaseId && (
        <div>
          <Button variant="outline" size="sm" asChild>
            <Link href={`/diseases/${encodeURIComponent(diagnosis.diseaseId)}`}>
              Open in disease library
              <ArrowRightIcon data-icon="inline-end" />
            </Link>
          </Button>
        </div>
      )}
    </div>
  )
}

function UncertainVerdict({ diagnosis }: { diagnosis: Diagnosis }) {
  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="border-uncertain/35 text-uncertain bg-uncertain/10 inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs font-medium">
            <CircleHelpIcon className="size-3.5" />
            Uncertain
          </span>
        </div>
        <h2 className="font-display text-uncertain text-4xl leading-tight font-semibold text-balance">
          Not confident enough to call it
        </h2>
        <p className="text-muted-foreground max-w-prose text-sm leading-relaxed">
          The top prediction landed below the model&apos;s decision threshold,
          so no diagnosis is being reported. This is the model declining to
          guess, not a processing error.
        </p>
      </div>

      <ThresholdGauge
        confidence={diagnosis.confidence}
        threshold={diagnosis.threshold}
        tone="uncertain"
      />

      <Alternatives
        diagnosis={diagnosis}
        title="What it was leaning toward"
        caveat="Shown for transparency. None of these cleared the threshold, so none is a diagnosis."
      />

      <div className="border-border flex flex-col gap-2 border-t pt-5">
        <span className="label-micro">What helps</span>
        <ul className="text-muted-foreground flex list-disc flex-col gap-1.5 pl-4 text-sm">
          <li>Fill the frame with a single leaf, face on.</li>
          <li>Even light — avoid hard shadows and blown highlights.</li>
          <li>Plain background, so the leaf is the only subject.</li>
        </ul>
      </div>
    </div>
  )
}

function RejectedVerdict({ diagnosis }: { diagnosis: Diagnosis }) {
  return (
    <Alert>
      <ImageOffIcon />
      <AlertTitle>Photo rejected before analysis</AlertTitle>
      <AlertDescription>
        <p>{diagnosis.error ?? "The upload failed validation."}</p>
        <p>Retake the photo and upload it again.</p>
      </AlertDescription>
    </Alert>
  )
}

function FailedVerdict({ diagnosis }: { diagnosis: Diagnosis }) {
  return (
    <Alert variant="destructive">
      <TriangleAlertIcon />
      <AlertTitle>Analysis failed</AlertTitle>
      <AlertDescription>
        <p>{diagnosis.error ?? "Inference errored after retries."}</p>
        <p>The photo is still stored. Uploading it again will retry.</p>
      </AlertDescription>
    </Alert>
  )
}

/**
 * Ten of the 38 classes have no real field photographs behind them, only
 * studio plates. The backend flags those; hiding the flag would present the
 * weakest predictions with the same authority as the strongest.
 */
export function FieldValidationNotice() {
  return (
    <Alert>
      <FlaskConicalIcon />
      <AlertTitle>Trained on studio images only</AlertTitle>
      <AlertDescription>
        The model has never seen a real field photograph of this class. Treat
        this verdict as a prompt to inspect the plant, not as a finding.
      </AlertDescription>
    </Alert>
  )
}

function Alternatives({
  diagnosis,
  title,
  caveat,
}: {
  diagnosis: Diagnosis
  title: string
  caveat?: string
}) {
  const alternatives = diagnosis.alternatives ?? []
  if (!alternatives.length) return null

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1">
        <span className="label-micro">{title}</span>
        {caveat && (
          <p className="text-muted-foreground max-w-prose text-xs">{caveat}</p>
        )}
      </div>
      <ul className="flex flex-col">
        {alternatives.map((alternative, index) => {
          const parsed = parseRawLabel(alternative.rawLabel)
          return (
            <li key={alternative.rawLabel}>
              {index > 0 && <Separator />}
              <div className="flex items-center justify-between gap-4 py-2.5">
                <div className="flex min-w-0 flex-col gap-1">
                  <span className="truncate text-sm font-medium">
                    {alternative.condition || parsed.condition}
                  </span>
                  <RawLabel rawLabel={alternative.rawLabel} />
                </div>
                <span className="text-muted-foreground shrink-0 font-mono text-sm tabular-nums">
                  {formatConfidence(alternative.confidence)}
                </span>
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
