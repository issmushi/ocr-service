export interface FieldOcr {
  model: string;
  engine: string;
  device: string | null;
  status: "success" | "error" | "skipped";
  text: string | null;
  confidence: number | null;
  duration_seconds: number;
  error: { code: string; message: string } | null;
}

export interface FieldOcrSummary {
  model: string;
  engine: string;
  duration_seconds: number;
  processed_fields: number;
  failed_fields: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function isProbability(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

function isDuration(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

export function isFieldOcr(value: unknown): value is FieldOcr {
  if (!isRecord(value)) return false;
  return typeof value.model === "string" && Boolean(value.model.trim())
    && typeof value.engine === "string" && Boolean(value.engine.trim())
    && (value.device === null || typeof value.device === "string")
    && typeof value.status === "string" && ["success", "error", "skipped"].includes(value.status)
    && (value.text === null || typeof value.text === "string")
    && (value.confidence === null || isProbability(value.confidence))
    && isDuration(value.duration_seconds)
    && (value.error === null || (isRecord(value.error)
      && typeof value.error.code === "string" && typeof value.error.message === "string"));
}

export function isOcrSummary(value: unknown): value is FieldOcrSummary {
  if (!isRecord(value)) return false;
  return typeof value.model === "string" && Boolean(value.model.trim())
    && typeof value.engine === "string" && Boolean(value.engine.trim())
    && isDuration(value.duration_seconds)
    && typeof value.processed_fields === "number" && Number.isSafeInteger(value.processed_fields) && value.processed_fields >= 0
    && typeof value.failed_fields === "number" && Number.isSafeInteger(value.failed_fields) && value.failed_fields >= 0;
}
