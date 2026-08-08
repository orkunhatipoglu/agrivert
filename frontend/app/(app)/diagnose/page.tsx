"use client"

import * as React from "react"
import Link from "next/link"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import {
  ArrowRightIcon,
  CameraIcon,
  ImageIcon,
  RotateCcwIcon,
  TriangleAlertIcon,
  UploadIcon,
} from "lucide-react"

import { FeedbackForm } from "@/components/feedback-form"
import { PageHeader } from "@/components/page-header"
import { StatusBadge } from "@/components/status-badge"
import { VerdictPanel } from "@/components/verdict-panel"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Field, FieldDescription, FieldLabel } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import { Spinner } from "@/components/ui/spinner"
import { useDiagnosisLive } from "@/hooks/use-diagnosis-live"
import { asRejectedUpload, diagnosesApi } from "@/lib/api"
import { checkImageFile, formatBytes } from "@/lib/image-checks"
import { cn } from "@/lib/utils"

export default function DiagnosePage() {
  const queryClient = useQueryClient()

  const [file, setFile] = React.useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = React.useState<string | null>(null)
  const [localError, setLocalError] = React.useState<string | null>(null)
  const [plotId, setPlotId] = React.useState("")
  const [farmId, setFarmId] = React.useState("")
  const [diagnosisId, setDiagnosisId] = React.useState<string | null>(null)
  const [dragging, setDragging] = React.useState(false)

  const fileInputRef = React.useRef<HTMLInputElement>(null)
  const cameraInputRef = React.useRef<HTMLInputElement>(null)

  const { diagnosis, polling } = useDiagnosisLive(diagnosisId)

  // The preview URL is created alongside the file rather than in an effect, so
  // there's no render where a chosen file has no preview. The ref mirrors it
  // purely so the unmount cleanup can revoke whatever is current.
  const previewRef = React.useRef<string | null>(null)

  React.useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    }
  }, [])

  function setSelection(next: File | null) {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    const url = next ? URL.createObjectURL(next) : null
    previewRef.current = url
    setPreviewUrl(url)
    setFile(next)
  }

  const upload = useMutation({
    mutationFn: () =>
      diagnosesApi.create({
        file: file!,
        plotId: plotId.trim() || undefined,
        farmId: farmId.trim() || undefined,
      }),
    onSuccess: (created) => {
      setDiagnosisId(created.diagnosisId)
      void queryClient.invalidateQueries({ queryKey: ["diagnoses"] })
    },
  })

  async function acceptFile(next: File | null | undefined) {
    setLocalError(null)
    upload.reset()
    if (!next) return

    const check = await checkImageFile(next)
    if (!check.ok) {
      setSelection(null)
      setLocalError(check.reason ?? "That photo can't be used.")
      return
    }
    setSelection(next)
  }

  function reset() {
    setSelection(null)
    setLocalError(null)
    setDiagnosisId(null)
    upload.reset()
    if (fileInputRef.current) fileInputRef.current.value = ""
    if (cameraInputRef.current) cameraInputRef.current.value = ""
  }

  const rejected = asRejectedUpload(upload.error)
  const uploadError = upload.error
    ? (rejected?.reason ?? (upload.error as Error).message)
    : null
  const pending =
    Boolean(diagnosisId) &&
    (!diagnosis || diagnosis.status === "queued" || diagnosis.status === "processing")

  return (
    <>
      <PageHeader
        eyebrow="Capture"
        title="Diagnose a plant"
        description="One leaf, filling the frame, in even light. The photo is analysed on the server and the verdict comes back here."
        actions={
          (file || diagnosisId) && (
            <Button variant="outline" onClick={reset}>
              <RotateCcwIcon data-icon="inline-start" />
              Start over
            </Button>
          )
        }
      />

      <div className="grid gap-8 lg:grid-cols-2 lg:gap-12">
        {/* ---- Capture surface ---- */}
        <section className="flex flex-col gap-5">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="sr-only"
            onChange={(event) => void acceptFile(event.target.files?.[0])}
          />
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            className="sr-only"
            onChange={(event) => void acceptFile(event.target.files?.[0])}
          />

          {previewUrl ? (
            <figure className="flex flex-col gap-3">
              <div
                className={cn(
                  "bg-muted relative aspect-square overflow-hidden rounded-xl border",
                  pending && "agv-scan"
                )}
              >
                {/* eslint-disable-next-line @next/next/no-img-element -- local object URL, nothing for next/image to optimise. */}
                <img
                  src={previewUrl}
                  alt={file?.name ?? "Selected plant photo"}
                  className="size-full object-cover"
                />
              </div>
              {file && (
                <figcaption className="text-muted-foreground flex items-center justify-between gap-3 font-mono text-xs">
                  <span className="truncate">{file.name}</span>
                  <span className="shrink-0">{formatBytes(file.size)}</span>
                </figcaption>
              )}
            </figure>
          ) : (
            <div
              onDragOver={(event) => {
                event.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault()
                setDragging(false)
                void acceptFile(event.dataTransfer.files?.[0])
              }}
              className={cn(
                "flex aspect-square flex-col items-center justify-center gap-5 rounded-xl border border-dashed p-8 text-center transition-colors",
                dragging ? "border-primary bg-primary/5" : "border-border"
              )}
            >
              <ImageIcon className="text-muted-foreground size-8" />
              <div className="flex flex-col gap-1.5">
                <p className="font-medium">Drop a photo here</p>
                <p className="text-muted-foreground text-sm">
                  JPEG, PNG or WebP · up to 12 MB · at least 64px
                </p>
              </div>
              <div className="flex flex-wrap justify-center gap-2">
                <Button
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <UploadIcon data-icon="inline-start" />
                  Choose file
                </Button>
                <Button
                  variant="outline"
                  onClick={() => cameraInputRef.current?.click()}
                >
                  <CameraIcon data-icon="inline-start" />
                  Take photo
                </Button>
              </div>
            </div>
          )}

          {localError && (
            <Alert variant="destructive">
              <TriangleAlertIcon />
              <AlertTitle>Can&apos;t use that photo</AlertTitle>
              <AlertDescription>{localError}</AlertDescription>
            </Alert>
          )}

          {file && !diagnosisId && (
            <>
              <Collapsible>
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" size="sm" className="self-start">
                    Tag this to a plot
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="flex flex-col gap-4 pt-4">
                  <Field>
                    <FieldLabel htmlFor="plotId">Plot ID</FieldLabel>
                    <Input
                      id="plotId"
                      value={plotId}
                      onChange={(event) => setPlotId(event.target.value)}
                      className="font-mono"
                    />
                  </Field>
                  <Field>
                    <FieldLabel htmlFor="farmId">Farm ID</FieldLabel>
                    <Input
                      id="farmId"
                      value={farmId}
                      onChange={(event) => setFarmId(event.target.value)}
                      className="font-mono"
                    />
                    <FieldDescription>
                      Both are optional and free-form for now — farm and plot
                      management isn&apos;t built on the API yet, but anything
                      you enter is stored on the diagnosis and filters history.
                    </FieldDescription>
                  </Field>
                </CollapsibleContent>
              </Collapsible>

              <Button
                size="lg"
                className="self-start"
                disabled={upload.isPending}
                onClick={() => upload.mutate()}
              >
                {upload.isPending && <Spinner />}
                Analyse this photo
              </Button>
            </>
          )}
        </section>

        {/* ---- Result ---- */}
        <section className="flex flex-col gap-6">
          {!diagnosisId && !uploadError && <HowItWorks />}

          {uploadError && (
            <Alert variant="destructive">
              <TriangleAlertIcon />
              <AlertTitle>
                {rejected ? "Photo rejected" : "Upload failed"}
              </AlertTitle>
              <AlertDescription>
                <p>{uploadError}</p>
                {rejected && <p>Retake the photo and try again.</p>}
              </AlertDescription>
            </Alert>
          )}

          {diagnosisId && (
            <>
              <div className="flex flex-wrap items-center gap-3">
                <StatusBadge status={diagnosis?.status ?? "queued"} />
                {polling && (
                  <span className="text-muted-foreground font-mono text-xs">
                    live stream unavailable — polling
                  </span>
                )}
              </div>

              {pending && (
                <div className="flex flex-col gap-2">
                  <p className="font-display text-2xl font-semibold">
                    {diagnosis?.status === "processing"
                      ? "Running inference"
                      : "Waiting for a worker"}
                  </p>
                  <p className="text-muted-foreground text-sm">
                    Preprocessing and classification run on the server. This
                    usually takes a few seconds.
                  </p>
                </div>
              )}

              {diagnosis && !pending && (
                <>
                  <VerdictPanel diagnosis={diagnosis} />
                  <Separator />
                  <FeedbackForm diagnosis={diagnosis} />
                  <div className="flex flex-wrap gap-2">
                    <Button variant="outline" size="sm" asChild>
                      <Link href={`/diagnoses/${diagnosis.diagnosisId}`}>
                        Open full record
                        <ArrowRightIcon data-icon="inline-end" />
                      </Link>
                    </Button>
                    <Button variant="ghost" size="sm" onClick={reset}>
                      Diagnose another
                    </Button>
                  </div>
                  {diagnosis.modelVersion && (
                    <p className="text-muted-foreground font-mono text-xs">
                      model {diagnosis.modelVersion}
                    </p>
                  )}
                </>
              )}
            </>
          )}
        </section>
      </div>
    </>
  )
}

/**
 * Ordered because the pipeline genuinely is a sequence — the numbers carry
 * information rather than decorating the list.
 */
function HowItWorks() {
  const steps = [
    {
      title: "Validate",
      body: "Type, size and dimensions are checked before anything is stored. A bad photo is rejected now, with a reason.",
    },
    {
      title: "Queue",
      body: "The photo goes to storage and the job to a worker, so a slow model never blocks the upload.",
    },
    {
      title: "Classify",
      body: "The worker applies the exact preprocessing used in training, then runs the active model.",
    },
    {
      title: "Decide",
      body: "Above the confidence threshold you get a verdict. Below it you get told the model isn't sure.",
    },
  ]

  return (
    <div className="flex flex-col gap-6">
      <span className="label-micro">What happens next</span>
      <ol className="flex flex-col gap-5">
        {steps.map((step, index) => (
          <li key={step.title} className="flex gap-4">
            <span className="text-muted-foreground mt-0.5 font-mono text-xs tabular-nums">
              {String(index + 1).padStart(2, "0")}
            </span>
            <div className="flex flex-col gap-1">
              <p className="text-sm font-medium">{step.title}</p>
              <p className="text-muted-foreground max-w-prose text-sm">
                {step.body}
              </p>
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}
