"use client"

import * as React from "react"
import Link from "next/link"
import { useParams, useRouter } from "next/navigation"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { format } from "date-fns"
import { ArrowLeftIcon, Trash2Icon, TriangleAlertIcon } from "lucide-react"
import { toast } from "sonner"

import { DiagnosisImage } from "@/components/diagnosis-image"
import { FeedbackForm } from "@/components/feedback-form"
import { RawLabel } from "@/components/raw-label"
import { StatusBadge } from "@/components/status-badge"
import { VerdictPanel } from "@/components/verdict-panel"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { useDiagnosisLive } from "@/hooks/use-diagnosis-live"
import { diagnosesApi } from "@/lib/api"
import type { Diagnosis } from "@/lib/types"

export default function DiagnosisDetailPage() {
  const params = useParams<{ id: string }>()
  const router = useRouter()
  const queryClient = useQueryClient()
  const id = params.id

  // Live, not static: opening a record that is still queued should resolve in
  // place rather than needing a refresh.
  const { diagnosis, isLoading, error, settled } = useDiagnosisLive(id)

  const remove = useMutation({
    mutationFn: () => diagnosesApi.remove(id),
    onSuccess: () => {
      toast.success("Diagnosis deleted.")
      void queryClient.invalidateQueries({ queryKey: ["diagnoses"] })
      router.push("/diagnoses")
    },
    onError: (caught: Error) => toast.error(caught.message),
  })

  if (isLoading) {
    return (
      <div className="grid gap-8 lg:grid-cols-2">
        <Skeleton className="aspect-square rounded-xl" />
        <div className="flex flex-col gap-4">
          <Skeleton className="h-8 w-40" />
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>
      </div>
    )
  }

  if (error || !diagnosis) {
    return (
      <>
        <BackLink />
        <Alert variant="destructive">
          <TriangleAlertIcon />
          <AlertTitle>Couldn&apos;t load this diagnosis</AlertTitle>
          <AlertDescription>
            {(error as Error | null)?.message ??
              "It may have been deleted, or it belongs to another account."}
          </AlertDescription>
        </Alert>
      </>
    )
  }

  const pending = !settled

  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <BackLink />
        <div className="flex items-center gap-2">
          <StatusBadge status={diagnosis.status} />
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="ghost" size="sm">
                <Trash2Icon data-icon="inline-start" />
                Delete
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete this diagnosis?</AlertDialogTitle>
                <AlertDialogDescription>
                  The record and the stored photo are both removed. This
                  can&apos;t be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Keep it</AlertDialogCancel>
                <AlertDialogAction
                  onClick={() => remove.mutate()}
                  disabled={remove.isPending}
                >
                  Delete
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>

      <div className="grid gap-8 lg:grid-cols-2 lg:gap-12">
        <div className="flex flex-col gap-4">
          {/* A rejected upload is never stored, so there is no photo to show. */}
          {diagnosis.status === "rejected" ? (
            <div className="bg-muted text-muted-foreground flex aspect-square items-center justify-center rounded-xl p-8 text-center text-sm">
              The photo failed validation and wasn&apos;t stored.
            </div>
          ) : (
            <DiagnosisImage
              diagnosisId={diagnosis.diagnosisId}
              className="aspect-square w-full"
              scanning={pending}
            />
          )}
          <Metadata diagnosis={diagnosis} />
        </div>

        <div className="flex flex-col gap-6">
          {pending ? (
            <div className="flex flex-col gap-2">
              <p className="font-display text-2xl font-semibold">
                {diagnosis.status === "processing"
                  ? "Running inference"
                  : "Waiting for a worker"}
              </p>
              <p className="text-muted-foreground text-sm">
                This page updates itself as the job progresses.
              </p>
            </div>
          ) : (
            <>
              <VerdictPanel diagnosis={diagnosis} />
              <Separator />
              <FeedbackForm diagnosis={diagnosis} />
            </>
          )}
        </div>
      </div>
    </>
  )
}

function BackLink() {
  return (
    <Button variant="ghost" size="sm" asChild className="-ml-2">
      <Link href="/diagnoses">
        <ArrowLeftIcon data-icon="inline-start" />
        History
      </Link>
    </Button>
  )
}

function Metadata({ diagnosis }: { diagnosis: Diagnosis }) {
  const rows: { label: string; value: React.ReactNode }[] = [
    {
      label: "Captured",
      value: format(new Date(diagnosis.createdAt), "d MMM yyyy, HH:mm"),
    },
  ]

  if (diagnosis.modelVersion) {
    rows.push({ label: "Model", value: diagnosis.modelVersion })
  }
  if (diagnosis.rawLabel) {
    rows.push({
      label: "Class",
      value: <RawLabel rawLabel={diagnosis.rawLabel} />,
    })
  }
  rows.push({ label: "ID", value: diagnosis.diagnosisId })

  return (
    <dl className="flex flex-col gap-2.5">
      {rows.map((row) => (
        <div
          key={row.label}
          className="flex items-baseline justify-between gap-4"
        >
          <dt className="label-micro">{row.label}</dt>
          <dd className="text-muted-foreground truncate font-mono text-xs">
            {row.value}
          </dd>
        </div>
      ))}
    </dl>
  )
}
