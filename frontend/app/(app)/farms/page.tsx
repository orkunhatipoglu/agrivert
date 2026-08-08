"use client"

import * as React from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  ChevronDownIcon,
  MapPinIcon,
  PencilIcon,
  PlusIcon,
  SproutIcon,
  Trash2Icon,
  TriangleAlertIcon,
} from "lucide-react"
import { toast } from "sonner"

import { PageHeader } from "@/components/page-header"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
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
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { Spinner } from "@/components/ui/spinner"
import { farmsApi } from "@/lib/api"
import type { Farm, Plot } from "@/lib/types"

/**
 * Farms and plots management.
 *
 * Scoping a diagnosis to a plot is what turns single verdicts into a trend
 * (ROUTES.md flaw #6) — the ids created here are what `/diagnoses` filters on.
 */
export default function FarmsPage() {
  const [createOpen, setCreateOpen] = React.useState(false)

  const {
    data,
    isLoading,
    error,
  } = useQuery({ queryKey: ["farms"], queryFn: farmsApi.list })

  return (
    <>
      <PageHeader
        eyebrow="Manage"
        title="Farms & plots"
        description="Scoping a diagnosis to a plot is what turns single verdicts into a trend you can act on."
        actions={
          <Button variant="outline" onClick={() => setCreateOpen(true)}>
            <PlusIcon data-icon="inline-start" />
            Add farm
          </Button>
        }
      />

      <FarmDialog open={createOpen} onOpenChange={setCreateOpen} />

      {isLoading ? (
        <div className="flex flex-col gap-3">
          {Array.from({ length: 3 }).map((_, index) => (
            <Skeleton key={index} className="h-24 rounded-lg" />
          ))}
        </div>
      ) : error ? (
        <Alert variant="destructive">
          <TriangleAlertIcon />
          <AlertTitle>Couldn&apos;t load farms</AlertTitle>
          <AlertDescription>{(error as Error).message}</AlertDescription>
        </Alert>
      ) : !data?.items.length ? (
        <Empty className="border-border/70 rounded-lg border border-dashed">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <SproutIcon />
            </EmptyMedia>
            <EmptyTitle>No farms yet</EmptyTitle>
            <EmptyDescription>
              Add a farm, then add the plots inside it. Diagnoses tagged with a
              plot become filterable history.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : (
        <div className="flex flex-col gap-3">
          {data.items.map((farm) => (
            <FarmCard key={farm.farmId} farm={farm} />
          ))}
        </div>
      )}
    </>
  )
}

function FarmCard({ farm }: { farm: Farm }) {
  const queryClient = useQueryClient()
  const [open, setOpen] = React.useState(false)
  const [editOpen, setEditOpen] = React.useState(false)
  const [addPlotOpen, setAddPlotOpen] = React.useState(false)

  // Plots load only once the card is expanded — a farmer with a dozen farms
  // shouldn't pay a dozen requests to see a list of names.
  const { data: plots, isLoading } = useQuery({
    queryKey: ["plots", farm.farmId],
    queryFn: () => farmsApi.listPlots(farm.farmId),
    enabled: open,
  })

  const remove = useMutation({
    mutationFn: () => farmsApi.remove(farm.farmId),
    onSuccess: () => {
      toast.success(`Deleted ${farm.name}.`)
      void queryClient.invalidateQueries({ queryKey: ["farms"] })
    },
    onError: (error: Error) => toast.error(error.message),
  })

  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
      className="rounded-lg border"
    >
      <div className="flex flex-wrap items-center gap-3 p-5">
        <CollapsibleTrigger asChild>
          <button className="flex flex-1 items-center gap-3 text-left">
            <ChevronDownIcon
              className={`text-muted-foreground size-4 shrink-0 transition-transform ${
                open ? "" : "-rotate-90"
              }`}
            />
            <span className="flex flex-col gap-1">
              <span className="font-display text-lg font-semibold">
                {farm.name}
              </span>
              <span className="text-muted-foreground flex items-center gap-2 text-xs">
                {farm.region ? (
                  <span className="flex items-center gap-1">
                    <MapPinIcon className="size-3" />
                    {farm.region}
                  </span>
                ) : (
                  <span>No region</span>
                )}
                <span className="font-mono">{farm.farmId.slice(0, 8)}</span>
              </span>
            </span>
          </button>
        </CollapsibleTrigger>

        <div className="flex shrink-0 gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setEditOpen(true)}
            aria-label={`Edit ${farm.name}`}
          >
            <PencilIcon />
          </Button>
          <DeleteButton
            label={`Delete ${farm.name}?`}
            description="The farm and all of its plots are removed. Diagnoses already filed against those plots are kept — they record something that was actually observed."
            pending={remove.isPending}
            onConfirm={() => remove.mutate()}
          />
        </div>
      </div>

      <CollapsibleContent>
        <Separator />
        <div className="flex flex-col gap-3 p-5">
          {isLoading ? (
            <Skeleton className="h-16 rounded-md" />
          ) : !plots?.items.length ? (
            <p className="text-muted-foreground text-sm">
              No plots in this farm yet.
            </p>
          ) : (
            <div className="flex flex-col gap-2">
              {plots.items.map((plot) => (
                <PlotRow key={plot.plotId} farm={farm} plot={plot} />
              ))}
            </div>
          )}

          <div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setAddPlotOpen(true)}
            >
              <PlusIcon data-icon="inline-start" />
              Add plot
            </Button>
          </div>
        </div>
      </CollapsibleContent>

      <FarmDialog farm={farm} open={editOpen} onOpenChange={setEditOpen} />
      <PlotDialog
        farmId={farm.farmId}
        open={addPlotOpen}
        onOpenChange={setAddPlotOpen}
      />
    </Collapsible>
  )
}

function PlotRow({ farm, plot }: { farm: Farm; plot: Plot }) {
  const queryClient = useQueryClient()
  const [editOpen, setEditOpen] = React.useState(false)

  const remove = useMutation({
    mutationFn: () => farmsApi.removePlot(farm.farmId, plot.plotId),
    onSuccess: () => {
      toast.success(`Deleted ${plot.name}.`)
      void queryClient.invalidateQueries({ queryKey: ["plots", farm.farmId] })
    },
    onError: (error: Error) => toast.error(error.message),
  })

  return (
    <div className="bg-muted/40 flex flex-wrap items-center gap-3 rounded-md p-3">
      <div className="flex flex-1 flex-col gap-1">
        <span className="text-sm font-medium">{plot.name}</span>
        <span className="text-muted-foreground flex flex-wrap items-center gap-2 text-xs">
          <Badge variant="secondary">{plot.cropType}</Badge>
          {plot.areaHectares != null && <span>{plot.areaHectares} ha</span>}
          {/* The id is shown because it's what you paste on the capture
              screen to tag a photo, and what history filters on. */}
          <code className="font-mono">{plot.plotId}</code>
        </span>
      </div>
      <div className="flex shrink-0 gap-1">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setEditOpen(true)}
          aria-label={`Edit ${plot.name}`}
        >
          <PencilIcon />
        </Button>
        <DeleteButton
          label={`Delete ${plot.name}?`}
          description="Diagnoses already tagged with this plot keep their plotId, so history is not rewritten."
          pending={remove.isPending}
          onConfirm={() => remove.mutate()}
        />
      </div>
      <PlotDialog
        farmId={farm.farmId}
        plot={plot}
        open={editOpen}
        onOpenChange={setEditOpen}
      />
    </div>
  )
}

function DeleteButton({
  label,
  description,
  pending,
  onConfirm,
}: {
  label: string
  description: string
  pending: boolean
  onConfirm: () => void
}) {
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button variant="ghost" size="sm" aria-label={label}>
          {pending ? <Spinner /> : <Trash2Icon />}
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{label}</AlertDialogTitle>
          <AlertDialogDescription>{description}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={onConfirm}>Delete</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

/** Create when `farm` is omitted, edit when it is given. */
function FarmDialog({
  farm,
  open,
  onOpenChange,
}: {
  farm?: Farm
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{farm ? "Edit farm" : "Add a farm"}</DialogTitle>
          <DialogDescription>
            A farm groups the plots you file diagnoses against.
          </DialogDescription>
        </DialogHeader>
        {/* Mounted only while open, so the fields initialise from `farm` on
            every open. Syncing them with an effect instead would cascade an
            extra render on each keystroke-free open. */}
        {open && <FarmForm farm={farm} onDone={() => onOpenChange(false)} />}
      </DialogContent>
    </Dialog>
  )
}

function FarmForm({ farm, onDone }: { farm?: Farm; onDone: () => void }) {
  const queryClient = useQueryClient()
  const [name, setName] = React.useState(farm?.name ?? "")
  const [region, setRegion] = React.useState(farm?.region ?? "")

  const save = useMutation({
    mutationFn: () => {
      const trimmed = name.trim()
      const body = { region: region.trim() || null }
      return farm
        ? farmsApi.update(farm.farmId, { ...body, name: trimmed })
        : farmsApi.create({ ...body, name: trimmed })
    },
    onSuccess: () => {
      toast.success(farm ? "Farm updated." : "Farm created.")
      void queryClient.invalidateQueries({ queryKey: ["farms"] })
      onDone()
    },
    onError: (error: Error) => toast.error(error.message),
  })

  return (
    <>
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
            placeholder="Optional"
          />
        </Field>
      </FieldGroup>
      <DialogFooter>
        <Button variant="ghost" onClick={onDone}>
          Cancel
        </Button>
        <Button
          disabled={!name.trim() || save.isPending}
          onClick={() => save.mutate()}
        >
          {save.isPending && <Spinner />}
          {farm ? "Save" : "Create farm"}
        </Button>
      </DialogFooter>
    </>
  )
}

/** Create when `plot` is omitted, edit when it is given. */
function PlotDialog({
  farmId,
  plot,
  open,
  onOpenChange,
}: {
  farmId: string
  plot?: Plot
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{plot ? "Edit plot" : "Add a plot"}</DialogTitle>
          <DialogDescription>
            The plot id is what tags a photo on the capture screen and what
            history filters on.
          </DialogDescription>
        </DialogHeader>
        {open && (
          <PlotForm
            farmId={farmId}
            plot={plot}
            onDone={() => onOpenChange(false)}
          />
        )}
      </DialogContent>
    </Dialog>
  )
}

function PlotForm({
  farmId,
  plot,
  onDone,
}: {
  farmId: string
  plot?: Plot
  onDone: () => void
}) {
  const queryClient = useQueryClient()
  const [name, setName] = React.useState(plot?.name ?? "")
  const [cropType, setCropType] = React.useState(plot?.cropType ?? "")
  const [area, setArea] = React.useState(
    plot?.areaHectares != null ? String(plot.areaHectares) : ""
  )

  const parsedArea = area.trim() ? Number(area) : null
  // The API rejects area <= 0, so catch it here rather than round-tripping.
  const areaInvalid =
    parsedArea !== null && (Number.isNaN(parsedArea) || parsedArea <= 0)

  const save = useMutation({
    mutationFn: () => {
      const body = {
        name: name.trim(),
        cropType: cropType.trim(),
        areaHectares: parsedArea,
      }
      return plot
        ? farmsApi.updatePlot(farmId, plot.plotId, body)
        : farmsApi.createPlot(farmId, body)
    },
    onSuccess: () => {
      toast.success(plot ? "Plot updated." : "Plot created.")
      void queryClient.invalidateQueries({ queryKey: ["plots", farmId] })
      onDone()
    },
    onError: (error: Error) => toast.error(error.message),
  })

  return (
    <>
      <FieldGroup>
        <Field>
          <FieldLabel htmlFor="plotName">Name</FieldLabel>
          <Input
            id="plotName"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
        </Field>
        <Field>
          <FieldLabel htmlFor="plotCrop">Crop type</FieldLabel>
          <Input
            id="plotCrop"
            value={cropType}
            onChange={(event) => setCropType(event.target.value)}
            placeholder="Tomato"
          />
        </Field>
        <Field>
          <FieldLabel htmlFor="plotArea">Area (hectares)</FieldLabel>
          <Input
            id="plotArea"
            type="number"
            min="0"
            step="0.1"
            value={area}
            onChange={(event) => setArea(event.target.value)}
            placeholder="Optional"
            aria-invalid={areaInvalid}
          />
        </Field>
      </FieldGroup>
      <DialogFooter>
        <Button variant="ghost" onClick={onDone}>
          Cancel
        </Button>
        <Button
          disabled={
            !name.trim() || !cropType.trim() || areaInvalid || save.isPending
          }
          onClick={() => save.mutate()}
        >
          {save.isPending && <Spinner />}
          {plot ? "Save" : "Create plot"}
        </Button>
      </DialogFooter>
    </>
  )
}
