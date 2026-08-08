/**
 * The model's class names are machine identifiers: `Tomato___Late_blight`,
 * `Pepper__bell___healthy`. Crop and condition are separated by a triple
 * underscore, and words within each by single underscores.
 *
 * These are shown verbatim in mono type wherever the exact class matters
 * (feedback corrections, admin, the KB), because a farmer correcting a verdict
 * has to name a class the model actually knows.
 */

export interface ParsedLabel {
  crop: string
  condition: string
  healthy: boolean
}

export function parseRawLabel(rawLabel: string): ParsedLabel {
  const [cropPart, ...rest] = rawLabel.split("___")
  const conditionPart = rest.join("___")
  return {
    crop: humanize(cropPart),
    condition: humanize(conditionPart || "unknown"),
    healthy: conditionPart.toLowerCase() === "healthy",
  }
}

function humanize(segment: string): string {
  const cleaned = segment
    .replace(/_+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
  if (!cleaned) return "Unknown"
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1)
}

/** `0.9312` -> `93.1%`. Confidence is always shown to one decimal. */
export function formatConfidence(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—"
  return `${(value * 100).toFixed(1)}%`
}

/** Metric keys in metadata.json are snake_case; make them readable. */
export function formatMetricKey(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

export function formatMetricValue(value: number | string | null): string {
  if (value === null || value === undefined) return "—"
  if (typeof value === "string") return value
  // Metrics land in [0,1] (accuracy, f1) or are counts / losses.
  if (value > 0 && value <= 1) return `${(value * 100).toFixed(1)}%`
  return String(Number.isInteger(value) ? value : value.toFixed(4))
}
