"use client"

import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { PlusIcon, TriangleAlertIcon } from "lucide-react"
import { toast } from "sonner"

import { NotImplemented, isNotImplemented } from "@/components/not-implemented"
import { PageHeader } from "@/components/page-header"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { Spinner } from "@/components/ui/spinner"
import { farmsApi } from "@/lib/api"

/**
 * Farms and plots exist as routes, schemas and auth on the API; the handlers
 * return 501. The UI is wired to the real calls rather than mocked, so what
 * you see here is exactly what the backend answers today.
 */
export default function FarmsPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["farms"],
    queryFn: farmsApi.list,
  })

  return (
    <>
      <PageHeader
        eyebrow="Manage"
        title="Farms & plots"
        description="Scoping a diagnosis to a plot is what turns single verdicts into a trend you can act on."
        actions={<CreateFarmDialog />}
      />

      {isLoading ? (
        <div className="flex flex-col gap-3">
          {Array.from({ length: 3 }).map((_, index) => (
            <Skeleton key={index} className="h-24 rounded-lg" />
          ))}
        </div>
      ) : isNotImplemented(error) ? (
        <>
          <NotImplemented route="GET /api/v1/farms" error={error} />
          <PlannedShape />
        </>
      ) : error ? (
        <Alert variant="destructive">
          <TriangleAlertIcon />
          <AlertTitle>Couldn&apos;t load farms</AlertTitle>
          <AlertDescription>{(error as Error).message}</AlertDescription>
        </Alert>
      ) : (
        <div className="flex flex-col gap-3">
          {data?.items.map((farm) => (
            <article
              key={farm.farmId}
              className="flex flex-col gap-2 rounded-lg border p-5"
            >
              <h2 className="font-display text-lg font-semibold">{farm.name}</h2>
              <p className="text-muted-foreground font-mono text-xs">
                {farm.region ?? "No region"} · {farm.farmId}
              </p>
            </article>
          ))}
        </div>
      )}
    </>
  )
}

function CreateFarmDialog() {
  const [open, setOpen] = React.useState(false)
  const [name, setName] = React.useState("")
  const [region, setRegion] = React.useState("")

  const create = useMutation({
    mutationFn: () => farmsApi.create({ name, region: region || undefined }),
    onSuccess: () => {
      toast.success("Farm created.")
      setOpen(false)
    },
    onError: (error: Error) => toast.error(error.message),
  })

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline">
          <PlusIcon data-icon="inline-start" />
          Add farm
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add a farm</DialogTitle>
          <DialogDescription>
            This posts to <code className="font-mono text-xs">POST /farms</code>
            , which the API hasn&apos;t implemented yet — the response will say
            so.
          </DialogDescription>
        </DialogHeader>
        <FieldGroup>
          <Field>
            <FieldLabel htmlFor="farmName">Name</FieldLabel>
            <Input
              id="farmName"
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </Field>
          <Field>
            <FieldLabel htmlFor="farmRegion">Region</FieldLabel>
            <Input
              id="farmRegion"
              value={region}
              onChange={(event) => setRegion(event.target.value)}
            />
          </Field>
        </FieldGroup>
        <DialogFooter>
          <Button variant="ghost" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button
            disabled={!name.trim() || create.isPending}
            onClick={() => create.mutate()}
          >
            {create.isPending && <Spinner />}
            Create farm
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/** What the API already agrees on, so it's clear this isn't a blank slate. */
function PlannedShape() {
  const routes = [
    { method: "GET", path: "/farms", note: "List your farms" },
    { method: "POST", path: "/farms", note: "Create a farm" },
    { method: "GET", path: "/farms/{farmId}", note: "Farm details" },
    { method: "PATCH", path: "/farms/{farmId}", note: "Rename, set region" },
    {
      method: "DELETE",
      path: "/farms/{farmId}",
      note: "Blocked on a cascade policy for plots and diagnoses",
    },
    { method: "GET", path: "/farms/{farmId}/plots", note: "List plots" },
    {
      method: "POST",
      path: "/farms/{farmId}/plots",
      note: "Create a plot with crop type and area",
    },
    { method: "PATCH", path: "/farms/{farmId}/plots/{plotId}", note: "Update" },
    { method: "DELETE", path: "/farms/{farmId}/plots/{plotId}", note: "Remove" },
  ]

  return (
    <section className="flex flex-col gap-4">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-display text-lg font-semibold">
          Already wired on the API
        </h2>
        <p className="text-muted-foreground max-w-prose text-sm">
          Diagnoses already carry <code className="font-mono text-xs">farmId</code>{" "}
          and <code className="font-mono text-xs">plotId</code>, and history
          filters on both — so you can tag and filter today by typing the IDs on
          the capture screen. Only the management CRUD is missing.
        </p>
      </div>
      <div className="rounded-xl border">
        {routes.map((route, index) => (
          <div key={`${route.method} ${route.path}`}>
            {index > 0 && <Separator />}
            <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 p-3.5">
              <span className="text-muted-foreground w-14 shrink-0 font-mono text-xs">
                {route.method}
              </span>
              <code className="font-mono text-xs">{route.path}</code>
              <span className="text-muted-foreground ml-auto text-xs">
                {route.note}
              </span>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
