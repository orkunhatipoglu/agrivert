/**
 * Client-side mirror of `backend/app/services/image_validation.py`.
 *
 * The backend is still the authority — it does a full decode to catch
 * truncated JPEGs, which a browser can't be trusted to replicate. This exists
 * so the obvious failures (wrong type, oversized, too small) are caught before
 * a farmer on a slow connection uploads 12MB to be told no.
 */

export const MAX_UPLOAD_BYTES = 12 * 1024 * 1024
export const MIN_IMAGE_DIMENSION = 64
export const ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]

export interface LocalCheckResult {
  ok: boolean
  reason?: string
  width?: number
  height?: number
}

export async function checkImageFile(file: File): Promise<LocalCheckResult> {
  if (file.size === 0) {
    return { ok: false, reason: "That file is empty." }
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    return {
      ok: false,
      reason: `Photo is ${(file.size / 1024 / 1024).toFixed(1)} MB; the limit is ${
        MAX_UPLOAD_BYTES / 1024 / 1024
      } MB.`,
    }
  }

  if (file.type && !ALLOWED_IMAGE_TYPES.includes(file.type.toLowerCase())) {
    return {
      ok: false,
      reason: `${file.type} isn't supported. Use JPEG, PNG or WebP.`,
    }
  }

  const dimensions = await readDimensions(file)
  if (!dimensions) {
    return { ok: false, reason: "That file couldn't be read as an image." }
  }

  if (Math.min(dimensions.width, dimensions.height) < MIN_IMAGE_DIMENSION) {
    return {
      ok: false,
      reason: `Photo is ${dimensions.width}×${dimensions.height}. The shortest side needs at least ${MIN_IMAGE_DIMENSION}px.`,
      ...dimensions,
    }
  }

  return { ok: true, ...dimensions }
}

function readDimensions(
  file: File
): Promise<{ width: number; height: number } | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file)
    const image = new Image()
    image.onload = () => {
      URL.revokeObjectURL(url)
      resolve({ width: image.naturalWidth, height: image.naturalHeight })
    }
    image.onerror = () => {
      URL.revokeObjectURL(url)
      resolve(null)
    }
    image.src = url
  })
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
