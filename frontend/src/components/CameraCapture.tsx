import { useCallback, useEffect, useRef, useState } from "react";
import { Icon } from "./Icon";
import { captureCameraFrames, waitForCameraFrame } from "./cameraBurst";

interface CameraCaptureProps {
  frameCount: 1 | 3;
  onCapture: (files: File[]) => void;
  onClose: () => void;
  onFallback: () => void;
}

type CameraStatus = "starting" | "ready" | "error";

function cameraErrorMessage(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError") {
      return "Доступ к камере запрещён. Разрешите его в настройках браузера или загрузите готовые фото.";
    }
    if (error.name === "NotFoundError") {
      return "Камера не найдена на этом устройстве.";
    }
    if (error.name === "NotReadableError") {
      return "Камера занята другим приложением или недоступна.";
    }
  }
  return "Не удалось открыть камеру. Можно выбрать готовые фото или воспользоваться системной камерой.";
}

export function CameraCapture({ frameCount, onCapture, onClose, onFallback }: CameraCaptureProps) {
  const [status, setStatus] = useState<CameraStatus>("starting");
  const [errorMessage, setErrorMessage] = useState("");
  const [capturing, setCapturing] = useState(false);
  const [capturedCount, setCapturedCount] = useState(0);
  const [flash, setFlash] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const lifecycleRef = useRef<AbortController | null>(null);
  const captureRef = useRef<AbortController | null>(null);
  const flashTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const isBurst = frameCount === 3;

  const stopCamera = useCallback(() => {
    lifecycleRef.current?.abort();
    captureRef.current?.abort();
    captureRef.current = null;
    clearTimeout(flashTimerRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const closeCamera = useCallback(() => {
    stopCamera();
    onClose();
  }, [onClose, stopCamera]);

  const openFallback = () => {
    closeCamera();
    onFallback();
  };

  useEffect(() => {
    const lifecycle = new AbortController();
    lifecycleRef.current = lifecycle;
    setStatus("starting");

    const start = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
        });
        if (lifecycle.signal.aborted) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        const video = videoRef.current;
        if (!video) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        video.srcObject = stream;
        // Frame readiness is the source of truth; play() alone can remain pending.
        void video.play().catch(() => undefined);
        await waitForCameraFrame(video, lifecycle.signal, { timeoutMs: 15_000 });
        if (!lifecycle.signal.aborted) setStatus("ready");
      } catch (error) {
        if (lifecycle.signal.aborted) return;
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        setErrorMessage(cameraErrorMessage(error));
        setStatus("error");
      }
    };

    void start();
    return stopCamera;
  }, [stopCamera]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeCamera();
        return;
      }
      if (event.key !== "Tab") return;
      const dialog = dialogRef.current;
      if (!dialog) return;
      const buttons = Array.from(dialog.querySelectorAll<HTMLButtonElement>("button:not(:disabled)"));
      const first = buttons[0];
      const last = buttons.at(-1);
      if (!first || !last) return;
      const active = document.activeElement;
      if (!dialog.contains(active) || (event.shiftKey ? active === first : active === last)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      }
    };
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [closeCamera]);

  const takePhotos = async () => {
    const video = videoRef.current;
    const lifecycle = lifecycleRef.current;
    // The ref closes the gap before React applies the disabled button state.
    if (!video || !lifecycle || lifecycle.signal.aborted || status !== "ready" || captureRef.current) return;
    const capture = new AbortController();
    captureRef.current = capture;
    setCapturing(true);
    setCapturedCount(0);
    setErrorMessage("");
    let files: File[] | undefined;

    try {
      files = await captureCameraFrames(video, frameCount, capture.signal, (count) => {
        if (videoRef.current !== video || capture.signal.aborted || lifecycle.signal.aborted) return;
        setCapturedCount(count);
        setFlash(true);
        clearTimeout(flashTimerRef.current);
        flashTimerRef.current = setTimeout(() => {
          if (videoRef.current === video && !lifecycle.signal.aborted) setFlash(false);
        }, 160);
      });
    } catch (error) {
      if (videoRef.current === video && !capture.signal.aborted && !lifecycle.signal.aborted) {
        setCapturedCount(0);
        setErrorMessage(error instanceof Error ? error.message : "Не удалось снять фото. Повторите съёмку.");
      }
    } finally {
      if (captureRef.current === capture) captureRef.current = null;
      if (videoRef.current === video && !capture.signal.aborted && !lifecycle.signal.aborted) setCapturing(false);
    }

    if (files && videoRef.current === video && !capture.signal.aborted && !lifecycle.signal.aborted) onCapture(files);
  };

  return (
    <div
      aria-label="Съёмка документа"
      aria-modal="true"
      className="camera-modal"
      ref={dialogRef}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) closeCamera();
      }}
      role="dialog"
    >
      <div className="camera-dialog">
        <header>
          <div>
            <span className="eyebrow">Камера</span>
            <strong>Поместите документ целиком в рамку</strong>
          </div>
          <button aria-label="Закрыть камеру" autoFocus onClick={closeCamera} type="button">
            <Icon name="x" size={18} />
          </button>
        </header>

        <div className={`camera-viewport ${status === "error" ? "camera-viewport--error" : ""}`}>
          <video aria-label="Изображение с камеры" autoPlay muted playsInline ref={videoRef} />
          {status === "starting" && (
            <div className="camera-state" role="status">
              <Icon className="spin" name="refresh" size={25} />
              <strong>Подключаем камеру</strong>
            </div>
          )}
          {status === "error" && (
            <div className="camera-state camera-state--error" role="alert">
              <Icon name="info" size={25} />
              <strong>Камера недоступна</strong>
              <p>{errorMessage}</p>
              {isBurst && <p>Автоматическая серия требует доступа к камере. Вместо неё можно загрузить 3 готовых фото.</p>}
              <button onClick={openFallback} type="button">
                {isBurst ? "Загрузить 3 фото" : "Открыть системную камеру"}
              </button>
            </div>
          )}
          {status !== "error" && <div aria-hidden="true" className="camera-frame"><i /><i /><i /><i /></div>}
          {flash && <div aria-hidden="true" className="camera-flash" />}
        </div>

        {status === "ready" && errorMessage && (
          <p className="camera-capture-error" role="alert">
            {errorMessage} {isBurst && "Серия не сохранена. Снимите все 3 фото заново."}
          </p>
        )}

        <footer>
          <div>
            <strong aria-live="polite" role="status">
              {capturing ? `Снято: ${capturedCount} из ${frameCount}` : isBurst ? "3 фото за одно нажатие" : "Один снимок"}
            </strong>
            <span>{isBurst ? "Не двигайте камеру: 3 фото с интервалом не менее 0,35 с." : "Держите документ ровно и избегайте бликов."}</span>
            {status !== "error" && (
              <button className="camera-fallback" onClick={openFallback} type="button">
                {isBurst ? "Или загрузить 3 готовых фото" : "Системная камера / файл"}
              </button>
            )}
          </div>
          <button
            className="camera-shutter"
            disabled={status !== "ready" || capturing}
            onClick={() => void takePhotos()}
            type="button"
          >
            <span><Icon name="camera" size={21} /></span>
            {capturing ? `Съёмка ${capturedCount} из ${frameCount}…` : isBurst ? "Снять серию из 3 фото" : "Снять фото"}
          </button>
        </footer>
      </div>
    </div>
  );
}
