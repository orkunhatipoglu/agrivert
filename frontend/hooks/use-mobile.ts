import * as React from "react"

const MOBILE_BREAKPOINT = 768
const QUERY = `(max-width: ${MOBILE_BREAKPOINT - 1}px)`

function subscribe(onChange: () => void) {
  const mql = window.matchMedia(QUERY)
  mql.addEventListener("change", onChange)
  return () => mql.removeEventListener("change", onChange)
}

/**
 * `useSyncExternalStore` rather than state-plus-effect: a media query is an
 * external store, and reading it this way avoids the extra render (and the
 * momentary wrong value) that setting state from an effect produces.
 */
export function useIsMobile() {
  return React.useSyncExternalStore(
    subscribe,
    () => window.matchMedia(QUERY).matches,
    // No viewport on the server; match the sidebar's desktop-first default.
    () => false
  )
}
