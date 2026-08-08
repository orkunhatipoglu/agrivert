"use client"

import * as React from "react"

import { formatConfidence } from "@/lib/labels"
import { cn } from "@/lib/utils"

type Tone = "healthy" | "diseased" | "uncertain" | "neutral"

const TONE_FILL: Record<Tone, string> = {
  healthy: "bg-healthy",
  diseased: "bg-diseased",
  uncertain: "bg-uncertain",
  neutral: "bg-muted-foreground",
}

const TONE_TEXT: Record<Tone, string> = {
  healthy: "text-healthy",
  diseased: "text-diseased",
  uncertain: "text-uncertain",
  neutral: "text-muted-foreground",
}

/**
 * The decision instrument.
 *
 * The backend refuses to name a disease when confidence lands below the
 * model's threshold — that is the single most important behaviour in the
 * system, and a bare percentage hides it. This draws the threshold as a
 * physical gate the prediction either clears or doesn't, so "uncertain" reads
 * as a deliberate decision rather than a missing result.
 */
export function ThresholdGauge({
  confidence,
  threshold,
  tone = "neutral",
  size = "default",
  className,
}: {
  confidence: number | null | undefined
  threshold: number | null | undefined
  tone?: Tone
  size?: "default" | "compact"
  className?: string
}) {
  const hasValue = typeof confidence === "number" && !Number.isNaN(confidence)
  const value = hasValue ? Math.min(Math.max(confidence, 0), 1) : 0
  const gate =
    typeof threshold === "number" && !Number.isNaN(threshold)
      ? Math.min(Math.max(threshold, 0), 1)
      : null
  const clears = gate !== null && hasValue ? value >= gate : null
  const compact = size === "compact"

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {!compact && (
        <div className="flex items-baseline justify-between gap-4">
          <span className="label-micro">Confidence</span>
          <span
            className={cn(
              "font-mono text-2xl leading-none font-medium tabular-nums",
              TONE_TEXT[tone]
            )}
          >
            {formatConfidence(confidence)}
          </span>
        </div>
      )}

      <div
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={hasValue ? Number((value * 100).toFixed(1)) : undefined}
        aria-valuetext={
          hasValue
            ? `${formatConfidence(confidence)} confidence, ${
                gate === null
                  ? "no threshold recorded"
                  : clears
                    ? `above the ${formatConfidence(gate)} threshold`
                    : `below the ${formatConfidence(gate)} threshold`
              }`
            : "No confidence recorded"
        }
        aria-label="Prediction confidence against the model's decision threshold"
        className={cn(
          "bg-muted relative w-full overflow-hidden rounded-full",
          compact ? "h-1.5" : "h-3"
        )}
      >
        <div
          className={cn(
            "h-full rounded-full transition-[width] duration-700 ease-out",
            TONE_FILL[tone]
          )}
          style={{ width: `${value * 100}%` }}
        />
        {gate !== null && (
          <div
            className="bg-foreground/70 absolute inset-y-0 w-0.5"
            style={{ left: `${gate * 100}%` }}
          />
        )}
      </div>

      {gate !== null && !compact && (
        <div className="text-muted-foreground flex items-center justify-between gap-4 font-mono text-[0.6875rem]">
          <span>
            {clears
              ? "Cleared the decision threshold"
              : "Below the decision threshold"}
          </span>
          <span className="tabular-nums">
            threshold {formatConfidence(gate)}
          </span>
        </div>
      )}
    </div>
  )
}
