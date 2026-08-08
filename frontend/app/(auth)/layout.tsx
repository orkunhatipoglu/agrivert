import { BrandMark, BrandWordmark } from "@/components/brand"

/**
 * The thesis panel. What makes this system worth showing is not that it
 * classifies leaves — plenty do — but that it refuses to guess when it isn't
 * sure. That claim, with the real numbers behind it, is the first thing
 * anyone sees.
 */
export default function AuthLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="grid min-h-svh lg:grid-cols-[1.1fr_1fr]">
      <aside className="bg-sidebar text-sidebar-foreground relative hidden flex-col justify-between overflow-hidden p-12 lg:flex">
        {/* The LED spectrum a grow rack actually runs: red + blue. */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 opacity-70"
          style={{
            backgroundImage:
              "radial-gradient(70% 55% at 15% 0%, color-mix(in oklch, var(--primary) 26%, transparent), transparent 70%), radial-gradient(60% 50% at 95% 100%, color-mix(in oklch, var(--healthy) 18%, transparent), transparent 70%)",
          }}
        />

        <div className="relative flex items-center gap-2.5">
          <BrandMark className="text-primary size-7" />
          <BrandWordmark className="text-xl" />
        </div>

        <div className="relative flex max-w-lg flex-col gap-8">
          <h1
            className="font-display text-[clamp(2.25rem,3.6vw,3.25rem)] leading-[1.04] font-semibold tracking-tight text-balance"
            style={{ fontVariationSettings: '"wdth" 88' }}
          >
            Most plant scanners tell you something.
            <span className="text-primary"> This one tells you when it doesn&apos;t know.</span>
          </h1>
          <p className="text-sidebar-foreground/75 text-base leading-relaxed text-pretty">
            Photograph a leaf on any rack layer. The model classifies it against
            38 PlantVillage classes — and when confidence lands below the
            decision threshold, it withholds the verdict instead of inventing
            one.
          </p>

          <dl className="border-sidebar-border grid grid-cols-3 gap-6 border-t pt-6">
            {[
              { value: "65.3%", label: "field accuracy" },
              { value: "0.95", label: "decision threshold" },
              { value: "38", label: "classes" },
            ].map((stat) => (
              <div key={stat.label} className="flex flex-col gap-1.5">
                <dt className="font-mono text-2xl leading-none font-medium tabular-nums">
                  {stat.value}
                </dt>
                <dd className="text-sidebar-foreground/60 font-mono text-[0.6875rem] tracking-[0.12em] uppercase">
                  {stat.label}
                </dd>
              </div>
            ))}
          </dl>
        </div>

        <p className="text-sidebar-foreground/50 relative max-w-lg font-mono text-xs leading-relaxed">
          Field accuracy is measured on real photographs, not studio plates.
          The gap between the two is why the threshold exists.
        </p>
      </aside>

      <main className="flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-sm">
          <div className="mb-10 flex items-center gap-2.5 lg:hidden">
            <BrandMark className="text-primary size-6" />
            <BrandWordmark />
          </div>
          {children}
        </div>
      </main>
    </div>
  )
}
