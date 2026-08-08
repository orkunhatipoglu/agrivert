"use client"

import { cn } from "@/lib/utils"

/**
 * A model class, shown as the machine identifier it is.
 *
 * `Tomato___Late_blight` is not a typo to be tidied away: it is the exact
 * string the feedback endpoint validates against, and the one an operator
 * needs when correcting a verdict. The triple underscore separating crop from
 * condition is dimmed rather than removed, so the structure reads without the
 * identifier being altered.
 */
export function RawLabel({
  rawLabel,
  className,
}: {
  rawLabel: string
  className?: string
}) {
  const separator = rawLabel.includes("___") ? "___" : null
  const [crop, ...rest] = rawLabel.split("___")
  const condition = rest.join("___")

  return (
    <code
      className={cn(
        "bg-muted/60 text-muted-foreground rounded px-1.5 py-0.5 font-mono text-xs break-all",
        className
      )}
    >
      {separator ? (
        <>
          <span className="text-foreground/80">{crop}</span>
          <span className="opacity-40">{separator}</span>
          <span className="text-foreground/80">{condition}</span>
        </>
      ) : (
        rawLabel
      )}
    </code>
  )
}
