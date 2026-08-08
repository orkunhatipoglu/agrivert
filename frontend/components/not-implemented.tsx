"use client"

import { ConstructionIcon } from "lucide-react"

import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty"
import { ApiError } from "@/lib/api"

/**
 * The backend answers 501 for `/farms/*`, `/notifications/*` and
 * `/admin/stats`: the routes, schemas and auth are wired, the handlers are
 * not. Showing that plainly matters — an empty table here would read as
 * "you have no farms", which is a different and false statement.
 *
 * The 501 detail is written by the backend and explains what each route still
 * needs, so it is surfaced verbatim rather than replaced with generic copy.
 */
export function NotImplemented({
  route,
  error,
}: {
  route: string
  error?: unknown
}) {
  const detail =
    error instanceof ApiError && error.isNotImplemented ? error.message : null

  return (
    <Empty className="border-border/70 rounded-lg border border-dashed">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <ConstructionIcon />
        </EmptyMedia>
        <EmptyTitle>Not built yet</EmptyTitle>
        <EmptyDescription>
          <code className="font-mono text-xs">{route}</code> is scaffolded in the
          API — route, schema and auth are wired, the handler isn&apos;t.
        </EmptyDescription>
      </EmptyHeader>
      {detail && (
        <EmptyContent>
          <p className="text-muted-foreground max-w-prose text-sm text-balance">
            {detail}
          </p>
        </EmptyContent>
      )}
    </Empty>
  )
}

/** True when a query failed only because the handler is a 501 stub. */
export function isNotImplemented(error: unknown): boolean {
  return error instanceof ApiError && error.isNotImplemented
}
