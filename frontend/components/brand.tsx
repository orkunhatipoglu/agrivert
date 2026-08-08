import { cn } from "@/lib/utils"

/**
 * The mark: three stacked rack layers with a lens on the middle one — the
 * physical thing this software runs on. A modular shelf with a camera per
 * layer, drawn at the smallest size that still reads.
 */
export function BrandMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      className={cn("size-6", className)}
    >
      <rect x="2" y="3.5" width="20" height="3" rx="1" fill="currentColor" opacity="0.4" />
      <rect x="2" y="10.25" width="20" height="3.5" rx="1" fill="currentColor" />
      <rect x="2" y="17.5" width="20" height="3" rx="1" fill="currentColor" opacity="0.4" />
      <circle cx="12" cy="12" r="1.15" className="fill-background" />
    </svg>
  )
}

export function BrandWordmark({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "font-display text-[1.0625rem] leading-none font-semibold tracking-tight",
        className
      )}
      style={{ fontVariationSettings: '"wdth" 90' }}
    >
      Agrivert
    </span>
  )
}
