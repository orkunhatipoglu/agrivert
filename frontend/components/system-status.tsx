"use client"

import { useQuery } from "@tanstack/react-query"

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { healthApi } from "@/lib/api"
import type { HealthResponse } from "@/lib/types"
import { cn } from "@/lib/utils"

const DEPENDENCIES: {
  key: keyof Pick<
    HealthResponse,
    "modelReady" | "firestoreReady" | "brokerReady"
  >
  label: string
  consequence: string
}[] = [
  {
    key: "modelReady",
    label: "Model",
    consequence: "No version resolves, so new diagnoses will fail.",
  },
  {
    key: "firestoreReady",
    label: "Firestore",
    consequence: "Results can't be stored or read back.",
  },
  {
    key: "brokerReady",
    label: "Queue",
    consequence: "Uploads accept but never get picked up by a worker.",
  },
]

/**
 * `GET /health` reports dependency readiness, not just process liveness. It is
 * in the sidebar because every one of those dependencies being down changes
 * what the operator should expect from a photo they are about to take.
 */
export function SystemStatus() {
  const { data, isError } = useQuery({
    queryKey: ["health"],
    queryFn: healthApi.get,
    refetchInterval: 60_000,
    staleTime: 30_000,
    retry: false,
  })

  const state = isError || !data ? "unreachable" : data.status === "ok" ? "ok" : "degraded"

  const summary = {
    ok: "All systems ready",
    degraded: "Running degraded",
    unreachable: "API unreachable",
  }[state]

  const dot = {
    ok: "bg-healthy",
    degraded: "bg-uncertain",
    unreachable: "bg-destructive",
  }[state]

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <Popover>
          <PopoverTrigger asChild>
            <SidebarMenuButton tooltip={summary} className="font-mono text-xs">
              <span className="flex size-4 items-center justify-center">
                <span className={cn("size-2 rounded-full", dot)} />
              </span>
              <span className="truncate">
                {data?.modelVersion ?? summary}
              </span>
            </SidebarMenuButton>
          </PopoverTrigger>
          <PopoverContent side="top" align="start" className="w-72">
            <div className="flex flex-col gap-3">
              <div className="flex flex-col gap-1">
                <span className="label-micro">System status</span>
                <p className="text-sm font-medium">{summary}</p>
              </div>

              {state === "unreachable" ? (
                <p className="text-muted-foreground text-xs">
                  The browser can&apos;t reach{" "}
                  <code className="font-mono">/health</code>. Check that the API
                  is running and that{" "}
                  <code className="font-mono">NEXT_PUBLIC_API_BASE_URL</code>{" "}
                  points at it.
                </p>
              ) : (
                data && (
                  <>
                    <dl className="flex flex-col gap-2">
                      {DEPENDENCIES.map(({ key, label, consequence }) => (
                        <div key={key} className="flex flex-col gap-0.5">
                          <div className="flex items-center justify-between gap-3">
                            <dt className="text-sm">{label}</dt>
                            <dd
                              className={cn(
                                "font-mono text-xs",
                                data[key] ? "text-healthy" : "text-destructive"
                              )}
                            >
                              {data[key] ? "ready" : "down"}
                            </dd>
                          </div>
                          {!data[key] && (
                            <p className="text-muted-foreground text-xs">
                              {consequence}
                            </p>
                          )}
                        </div>
                      ))}
                    </dl>
                    <div className="text-muted-foreground flex flex-col gap-1 font-mono text-[0.6875rem]">
                      <span>env {data.environment}</span>
                      <span className="break-all">
                        model {data.modelVersion ?? "none"}
                      </span>
                    </div>
                    {data.detail && (
                      <p className="text-muted-foreground border-border border-t pt-2 font-mono text-[0.6875rem] break-words">
                        {data.detail}
                      </p>
                    )}
                  </>
                )
              )}
            </div>
          </PopoverContent>
        </Popover>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}
