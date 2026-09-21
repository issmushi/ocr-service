const FRAME_INTERVAL_MS = 350;
const FRAME_TIMEOUT_MS = 5_000;

interface FrameWaitOptions {
  previousMediaTime?: number;
  minDelayMs?: number;
  timeoutMs?: number;
}

function abortError(): DOMException {
  return new DOMException("Съёмка отменена.", "AbortError");
}

/** Wait for an actual video frame, and for the stream to advance between photos. */
export function waitForCameraFrame(
  video: HTMLVideoElement,
  signal: AbortSignal,
  { previousMediaTime, minDelayMs = 0, timeoutMs = FRAME_TIMEOUT_MS }: FrameWaitOptions = {},
): Promise<number> {
  return new Promise((resolve, reject) => {
    const notBefore = performance.now() + minDelayMs;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let deadline: ReturnType<typeof setTimeout> | undefined;
    let frameCallback: number | undefined;
    let settled = false;

    const cleanup = () => {
      clearTimeout(timer);
      clearTimeout(deadline);
      if (frameCallback !== undefined) video.cancelVideoFrameCallback(frameCallback);
      signal.removeEventListener("abort", onAbort);
    };
    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    };
    const onAbort = () => fail(abortError());
    const acceptFrame = (mediaTime: number): boolean => {
      if (
        video.readyState < 2 || video.videoWidth < 1 || video.videoHeight < 1 ||
        (previousMediaTime !== undefined && mediaTime <= previousMediaTime)
      ) return false;
      settled = true;
      cleanup();
      resolve(mediaTime);
      return true;
    };
    const wait = () => {
      if (signal.aborted) {
        onAbort();
        return;
      }
      const remainingDelay = notBefore - performance.now();
      if (remainingDelay > 0) {
        timer = setTimeout(wait, Math.ceil(remainingDelay));
        return;
      }
      if (typeof video.requestVideoFrameCallback === "function") {
        frameCallback = video.requestVideoFrameCallback((_now, metadata) => {
          frameCallback = undefined;
          if (!settled && !acceptFrame(metadata.mediaTime)) wait();
        });
      } else if (!acceptFrame(video.currentTime)) {
        timer = setTimeout(wait, 30);
      }
    };

    if (signal.aborted) {
      onAbort();
      return;
    }
    signal.addEventListener("abort", onAbort, { once: true });
    deadline = setTimeout(() => fail(new Error(
      "Камера перестала передавать новые кадры. Проверьте её и повторите съёмку.",
    )), minDelayMs + timeoutMs);
    wait();
  });
}

interface PhotoEncoderState {
  preferDataUrl: boolean;
}

function isImageBlob(blob: Blob | null): blob is Blob {
  return Boolean(blob && blob.size > 0 && /^(image\/jpeg|image\/png|image\/webp)$/.test(blob.type));
}

function dataUrlPhoto(canvas: HTMLCanvasElement): Blob {
  const dataUrl = canvas.toDataURL("image/jpeg", 0.94);
  const separator = dataUrl.indexOf(",");
  const mimeType = /^data:(image\/(?:jpeg|png|webp));base64$/.exec(dataUrl.slice(0, separator))?.[1];
  if (!mimeType || separator === dataUrl.length - 1) {
    throw new Error("Canvas toDataURL returned an empty or unsupported image.");
  }

  // Decode in small chunks; fetch(dataUrl) can be blocked by the page's CSP.
  const parts: Uint8Array[] = [];
  const chunkSize = 32_768; // A multiple of four keeps base64 chunk boundaries valid.
  for (let offset = separator + 1; offset < dataUrl.length; offset += chunkSize) {
    const binary = atob(dataUrl.slice(offset, offset + chunkSize));
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
    parts.push(bytes);
  }
  const blob = new Blob(parts, { type: mimeType });
  if (!isImageBlob(blob)) throw new Error("Canvas toDataURL returned an empty image.");
  return blob;
}

function photoEncodingError(errors: unknown[]): Error {
  const blocked = errors.some((error) => error instanceof DOMException && error.name === "SecurityError");
  return new Error(
    blocked
      ? "Браузер запретил сохранение фото. Загрузите готовые фото с устройства."
      : "Не удалось сохранить фото в этом браузере. Повторите съёмку или загрузите готовые фото.",
    { cause: new AggregateError(errors, "Camera photo encoding failed.") },
  );
}

function encodePhoto(
  canvas: HTMLCanvasElement,
  signal: AbortSignal,
  encoder: PhotoEncoderState,
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let usingFallback = false;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    const finish = (blob: Blob | null, error?: Error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      signal.removeEventListener("abort", onAbort);
      if (error) reject(error);
      else if (blob) resolve(blob);
    };
    const onAbort = () => finish(null, abortError());
    const fallback = (primaryError?: unknown) => {
      if (settled || usingFallback) return;
      if (signal.aborted) {
        onAbort();
        return;
      }
      usingFallback = true;
      clearTimeout(timeout);
      try {
        // The canvas still contains this exact frame; do not draw or resize it here.
        const blob = dataUrlPhoto(canvas);
        if (signal.aborted) {
          onAbort();
          return;
        }
        // Avoid waiting for an unavailable async encoder on the remaining photos.
        encoder.preferDataUrl = true;
        finish(blob);
      } catch (error) {
        if (signal.aborted) onAbort();
        else finish(null, photoEncodingError(primaryError === undefined ? [error] : [primaryError, error]));
      }
    };
    if (signal.aborted) {
      onAbort();
      return;
    }
    signal.addEventListener("abort", onAbort, { once: true });
    if (encoder.preferDataUrl) {
      fallback();
      return;
    }
    if (typeof canvas.toBlob !== "function") {
      fallback(new Error("Canvas toBlob is unavailable."));
      return;
    }
    // An absent callback is not proof of a bad image: use the retained pixels.
    timeout = setTimeout(() => fallback(new Error("Canvas toBlob did not finish within 5 seconds.")), FRAME_TIMEOUT_MS);
    try {
      canvas.toBlob((blob) => {
        if (settled || usingFallback) return;
        if (isImageBlob(blob)) finish(blob);
        else fallback(new Error("Canvas toBlob returned an empty or unsupported image."));
      }, "image/jpeg", 0.94);
    } catch (error) {
      fallback(error);
    }
  });
}

export async function captureCameraFrames(
  video: HTMLVideoElement,
  frameCount: 1 | 3,
  signal: AbortSignal,
  onFrameCaptured: (count: number) => void,
): Promise<File[]> {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Браузер не смог сформировать снимок. Повторите съёмку.");

  const files: File[] = [];
  const batchTimestamp = Date.now();
  let previousMediaTime: number | undefined;
  let previousCaptureTime: number | undefined;
  const encoder: PhotoEncoderState = { preferDataUrl: false };

  try {
    for (let index = 0; index < frameCount; index += 1) {
      previousMediaTime = await waitForCameraFrame(video, signal, {
        previousMediaTime,
        minDelayMs: previousCaptureTime === undefined
          ? 0
          : Math.max(0, FRAME_INTERVAL_MS - (performance.now() - previousCaptureTime)),
      });
      signal.throwIfAborted();
      // Reuse the backing bitmap while the camera resolution stays the same.
      if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
      if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      previousCaptureTime = performance.now();
      const blob = await encodePhoto(canvas, signal, encoder);
      signal.throwIfAborted();
      const extension = blob.type === "image/png" ? "png" : blob.type === "image/webp" ? "webp" : "jpg";
      files.push(new File([blob], `document-${batchTimestamp}-${index + 1}.${extension}`, {
        type: blob.type,
        lastModified: Date.now(),
      }));
      onFrameCaptured(files.length);
    }
    return files;
  } finally {
    // Release the full-resolution backing bitmap on success, cancellation or failure.
    canvas.width = 0;
    canvas.height = 0;
  }
}
