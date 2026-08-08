"use client"

import * as React from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"

import { diagnosesApi, streamDiagnosis } from "@/lib/api"
import { isTerminal, type Diagnosis } from "@/lib/types"

/**
 * Follow a diagnosis to its terminal state.
 *
 * Prefers the SSE stream, which only emits on change, and falls back to
 * polling `GET /diagnoses/{id}` if the stream errors or times out — the
 * backend closes the stream after 180s and explicitly tells clients to poll.
 * Both paths write into the same query cache entry, so the rest of the app
 * sees one source of truth.
 */
export function useDiagnosisLive(diagnosisId: string | null) {
  const queryClient = useQueryClient()
  // Which id the stream failed for, rather than a boolean that would need
  // resetting on every id change.
  const [streamFailedFor, setStreamFailedFor] = React.useState<string | null>(
    null
  )
  const streamFailed = Boolean(diagnosisId) && streamFailedFor === diagnosisId

  const query = useQuery({
    queryKey: ["diagnosis", diagnosisId],
    queryFn: () => diagnosesApi.get(diagnosisId!),
    enabled: Boolean(diagnosisId),
    // Poll only once the stream is unavailable, and only while unfinished.
    refetchInterval: (q) => {
      const data = q.state.data as Diagnosis | undefined
      if (!data || isTerminal(data.status)) return false
      return streamFailed ? 2000 : false
    },
  })

  const status = query.data?.status
  const settled = status ? isTerminal(status) : false

  React.useEffect(() => {
    if (!diagnosisId || settled) return

    const abort = streamDiagnosis(diagnosisId, {
      onStatus: (diagnosis) => {
        queryClient.setQueryData(["diagnosis", diagnosisId], diagnosis)
      },
      onError: () => setStreamFailedFor(diagnosisId),
      onClose: () => {
        // The stream ends at the terminal frame; refetch once so nothing is
        // missed if the last frame was dropped mid-flight.
        void queryClient.invalidateQueries({
          queryKey: ["diagnosis", diagnosisId],
        })
      },
    })

    return abort
    // `settled` is intentionally a dependency: once terminal, tear the stream
    // down rather than holding an idle connection open.
  }, [diagnosisId, settled, queryClient])

  return {
    diagnosis: query.data ?? null,
    isLoading: query.isLoading,
    error: query.error,
    settled,
    /** True when live push is unavailable and we've dropped back to polling. */
    polling: streamFailed && !settled,
  }
}
