"use client"

import * as React from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { CheckIcon, ThumbsDownIcon, ThumbsUpIcon } from "lucide-react"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Spinner } from "@/components/ui/spinner"
import { Textarea } from "@/components/ui/textarea"
import { diagnosesApi, diseasesApi } from "@/lib/api"
import type { Diagnosis } from "@/lib/types"

/**
 * The retraining corpus (ROUTES.md flaw #7).
 *
 * The backend validates a correction against the model's own class list and
 * rejects anything else, so corrections are picked from that list rather than
 * typed. Free text goes in the note, where it can't poison training labels.
 */
export function FeedbackForm({ diagnosis }: { diagnosis: Diagnosis }) {
  const queryClient = useQueryClient()
  const [agrees, setAgrees] = React.useState<boolean | null>(null)
  const [correctedRawLabel, setCorrectedRawLabel] = React.useState<string>("")
  const [note, setNote] = React.useState("")
  const [done, setDone] = React.useState(false)

  // Seeded from the active model's labels.json, so this is exactly the set the
  // feedback endpoint will accept.
  const { data: diseases } = useQuery({
    queryKey: ["diseases"],
    queryFn: diseasesApi.list,
    staleTime: 10 * 60_000,
    enabled: agrees === false,
  })

  const mutation = useMutation({
    mutationFn: () =>
      diagnosesApi.feedback(diagnosis.diagnosisId, {
        agrees: agrees!,
        correctedRawLabel: agrees ? null : correctedRawLabel,
        note: note.trim() || null,
      }),
    onSuccess: () => {
      setDone(true)
      toast.success("Thanks — that goes into the retraining set.")
      void queryClient.invalidateQueries({
        queryKey: ["diagnosis", diagnosis.diagnosisId],
      })
    },
    onError: (error: Error) => toast.error(error.message),
  })

  // The API rejects feedback on anything not yet decided.
  if (diagnosis.status !== "completed" && diagnosis.status !== "uncertain") {
    return null
  }

  if (done) {
    return (
      <div className="text-muted-foreground flex items-center gap-2 text-sm">
        <CheckIcon className="text-healthy size-4" />
        Feedback recorded.
      </div>
    )
  }

  const canSubmit =
    agrees === true || (agrees === false && correctedRawLabel !== "")

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-1.5">
        <span className="label-micro">Was this right?</span>
        <p className="text-muted-foreground max-w-prose text-sm">
          {diagnosis.status === "uncertain"
            ? "If you know what this plant has, telling us here is what teaches the model to stop hedging on it."
            : "Corrections are the only source of real field training data this model gets."}
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          variant={agrees === true ? "default" : "outline"}
          size="sm"
          onClick={() => setAgrees(true)}
        >
          <ThumbsUpIcon data-icon="inline-start" />
          {diagnosis.status === "uncertain" ? "Fair enough" : "That's correct"}
        </Button>
        <Button
          type="button"
          variant={agrees === false ? "default" : "outline"}
          size="sm"
          onClick={() => setAgrees(false)}
        >
          <ThumbsDownIcon data-icon="inline-start" />
          {diagnosis.status === "uncertain"
            ? "I know what it is"
            : "That's wrong"}
        </Button>
      </div>

      {agrees !== null && (
        <>
          <Separator />
          <FieldGroup>
            {agrees === false && (
              <Field>
                <FieldLabel htmlFor="corrected">Correct class</FieldLabel>
                <Select
                  value={correctedRawLabel}
                  onValueChange={setCorrectedRawLabel}
                >
                  <SelectTrigger id="corrected">
                    <SelectValue placeholder="Pick the class you saw" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="unknown">
                      I can&apos;t identify it
                    </SelectItem>
                    {diseases?.items.map((disease) => (
                      <SelectItem
                        key={disease.rawLabel}
                        value={disease.rawLabel}
                      >
                        {disease.crop} — {disease.condition}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <FieldDescription>
                  {diseases?.items.length
                    ? "Only classes the model knows can be accepted."
                    : "The disease library hasn't been seeded, so only “I can't identify it” is available."}
                </FieldDescription>
              </Field>
            )}

            <Field>
              <FieldLabel htmlFor="note">Note</FieldLabel>
              <Textarea
                id="note"
                rows={3}
                maxLength={2000}
                placeholder="Anything that would help someone reading this later."
                value={note}
                onChange={(event) => setNote(event.target.value)}
              />
              <FieldDescription>Optional.</FieldDescription>
            </Field>

            <Field>
              <Button
                type="button"
                size="sm"
                className="self-start"
                disabled={!canSubmit || mutation.isPending}
                onClick={() => mutation.mutate()}
              >
                {mutation.isPending && <Spinner />}
                Send feedback
              </Button>
            </Field>
          </FieldGroup>
        </>
      )}
    </div>
  )
}
