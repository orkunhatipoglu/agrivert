"use client"

import * as React from "react"
import { useQuery } from "@tanstack/react-query"
import { ImageOffIcon } from "lucide-react"

import { Skeleton } from "@/components/ui/skeleton"
import { diagnosesApi } from "@/lib/api"
import { cn } from "@/lib/utils"

/**
 * The stored photo for a diagnosis.
 *
 * `GET /diagnoses/{id}/image` requires a bearer token, which an `<img src>`
 * cannot send, so the bytes are fetched and turned into an object URL. The URL
 * is revoked when it changes or the component unmounts, or a long history
 * session would leak a blob per photo viewed.
 */
export function DiagnosisImage({
  diagnosisId,
  alt = "Uploaded plant photo",
  className,
  scanning = false,
}: {
  diagnosisId: string
  alt?: string
  className?: string
  /** Draws the inference sweep. Tied to a live job, never decorative. */
  scanning?: boolean
}) {
  const { data: blob, isError } = useQuery({
    queryKey: ["diagnosis-image", diagnosisId],
    queryFn: () => diagnosesApi.image(diagnosisId),
    staleTime: Infinity,
    retry: false,
  })

  const src = React.useMemo(
    () => (blob ? URL.createObjectURL(blob) : null),
    [blob]
  )

  React.useEffect(() => {
    if (!src) return
    return () => URL.revokeObjectURL(src)
  }, [src])

  if (isError) {
    return (
      <div
        className={cn(
          "bg-muted text-muted-foreground flex flex-col items-center justify-center gap-2 rounded-lg",
          className
        )}
      >
        <ImageOffIcon className="size-6" />
        <p className="text-xs">Photo is no longer stored</p>
      </div>
    )
  }

  if (!src) {
    return <Skeleton className={cn("rounded-lg", className)} />
  }

  return (
    <div
      className={cn(
        "bg-muted relative overflow-hidden rounded-lg",
        scanning && "agv-scan",
        className
      )}
    >
      {/* eslint-disable-next-line @next/next/no-img-element -- blob: URL from an authenticated fetch; next/image cannot optimise it. */}
      <img src={src} alt={alt} className="size-full object-cover" />
    </div>
  )
}
