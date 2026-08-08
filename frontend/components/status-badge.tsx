"use client"

import {
  CheckIcon,
  CircleAlertIcon,
  CircleHelpIcon,
  ClockIcon,
  LoaderIcon,
  TriangleAlertIcon,
  XIcon,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import type { DiagnosisStatus } from "@/lib/types"
import { cn } from "@/lib/utils"

/**
 * Copy note: each status says what happened, in the same words the rest of
 * the interface uses. "Uncertain" is not an error and is never styled like
 * one — the model declining to guess is the system working.
 */
const STATUS_META: Record<
  DiagnosisStatus,
  { label: string; icon: typeof CheckIcon; className: string }
> = {
  queued: {
    label: "Queued",
    icon: ClockIcon,
    className: "text-muted-foreground",
  },
  processing: {
    label: "Analysing",
    icon: LoaderIcon,
    className: "text-primary",
  },
  completed: {
    label: "Complete",
    icon: CheckIcon,
    className: "text-healthy",
  },
  uncertain: {
    label: "Uncertain",
    icon: CircleHelpIcon,
    className: "text-uncertain",
  },
  rejected: {
    label: "Rejected",
    icon: XIcon,
    className: "text-muted-foreground",
  },
  failed: {
    label: "Failed",
    icon: TriangleAlertIcon,
    className: "text-destructive",
  },
}

export function StatusBadge({
  status,
  className,
}: {
  status: DiagnosisStatus
  className?: string
}) {
  const meta = STATUS_META[status] ?? {
    label: status,
    icon: CircleAlertIcon,
    className: "text-muted-foreground",
  }
  const Icon = meta.icon

  return (
    <Badge variant="outline" className={cn("gap-1.5", className)}>
      <Icon
        className={cn(meta.className, status === "processing" && "animate-spin")}
      />
      {meta.label}
    </Badge>
  )
}

/**
 * The verdict itself — healthy vs diseased. Separate from status because a
 * `completed` diagnosis can be either, and a farmer scanning a list is
 * looking for this, not for job state.
 */
export function VerdictBadge({
  healthy,
  className,
}: {
  healthy: boolean | null | undefined
  className?: string
}) {
  if (healthy === null || healthy === undefined) return null

  return (
    <Badge
      variant="outline"
      className={cn(
        "gap-1.5 font-medium",
        healthy
          ? "border-healthy/35 text-healthy bg-healthy/10"
          : "border-diseased/35 text-diseased bg-diseased/10",
        className
      )}
    >
      {healthy ? <CheckIcon /> : <TriangleAlertIcon />}
      {healthy ? "Healthy" : "Diseased"}
    </Badge>
  )
}
