type Box = [number, number, number, number];
export type Quad = [[number, number], [number, number], [number, number], [number, number]];

interface CropImage {
  image: string;
  width: number;
  height: number;
  rotation_degrees: number;
}

export interface PassportFieldOcr {
  model: string;
  engine: string;
  device: string | null;
  status: "success" | "error" | "skipped";
  text: string | null;
  confidence: number | null;
  duration_seconds: number;
  error: { code: string; message: string } | null;
}

interface FieldOcrSummary {
  model: string;
  engine: string;
  duration_seconds: number;
  processed_fields: number;
  failed_fields: number;
}

export interface PassportFieldCrop extends CropImage {
  name: string;
  label: string;
  bbox: Box;
  detection_bbox: Box;
  confidence: number;
  ocr: PassportFieldOcr | null;
  kind?: "address_line" | "registration_stamp";
  quad?: Quad;
  crop_quad?: Quad;
}

export interface PassportFieldCrops {
  document: CropImage & { detected: boolean; orientation_confidence: number | null };
  fields: PassportFieldCrop[];
  durationSeconds: number | null;
  ocr: FieldOcrSummary | null;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

export function isProbability(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

function isDuration(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

export function isFieldOcr(value: unknown): value is PassportFieldOcr {
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

function isOcrSummary(value: unknown): value is FieldOcrSummary {
  if (!isRecord(value)) return false;
  return typeof value.model === "string" && Boolean(value.model.trim())
    && typeof value.engine === "string" && Boolean(value.engine.trim())
    && isDuration(value.duration_seconds)
    && typeof value.processed_fields === "number" && Number.isSafeInteger(value.processed_fields) && value.processed_fields >= 0
    && typeof value.failed_fields === "number" && Number.isSafeInteger(value.failed_fields) && value.failed_fields >= 0;
}

export function isCropImage(value: unknown, freeRotation = false): value is CropImage {
  if (!isRecord(value)) return false;
  // Only render embedded JPEG/PNG images; never request image URLs supplied by a server.
  return typeof value.image === "string" && /^data:image\/(?:jpeg|png);base64,[A-Za-z0-9+/]+={0,2}$/.test(value.image)
    && typeof value.width === "number" && Number.isSafeInteger(value.width) && value.width > 0
    && typeof value.height === "number" && Number.isSafeInteger(value.height) && value.height > 0
    && typeof value.rotation_degrees === "number" && Number.isFinite(value.rotation_degrees)
    && (freeRotation || [0, 90, 180, 270].includes(value.rotation_degrees));
}

function isBox(value: unknown, width: number, height: number): value is Box {
  return Array.isArray(value) && value.length === 4
    && value.every((coordinate) => typeof coordinate === "number" && Number.isFinite(coordinate))
    && value[0] >= 0 && value[1] >= 0
    && value[2] > value[0] && value[3] > value[1]
    && value[2] <= width && value[3] <= height;
}

export function parsePassportFieldCrops(value: unknown): PassportFieldCrops | null {
  if (!isRecord(value)) return null;
  const document = value.document;
  if (!isRecord(document) || !isCropImage(document) || typeof document.detected !== "boolean"
    || !(document.orientation_confidence === null || isProbability(document.orientation_confidence))
    || !Array.isArray(value.fields)) return null;

  const fields: PassportFieldCrop[] = [];
  for (const field of value.fields) {
    if (!isRecord(field) || !isCropImage(field)
      || typeof field.name !== "string" || !field.name.trim()
      || typeof field.label !== "string" || !field.label.trim()
      || !isBox(field.bbox, document.width, document.height)
      || !isBox(field.detection_bbox, document.width, document.height)
      || !isProbability(field.confidence)
      || (field.ocr != null && !isFieldOcr(field.ocr))) return null;
    fields.push({
      name: field.name, label: field.label, image: field.image,
      width: field.width, height: field.height, rotation_degrees: field.rotation_degrees,
      bbox: field.bbox, detection_bbox: field.detection_bbox, confidence: field.confidence,
      ocr: field.ocr ?? null,
    });
  }

  const duration = isRecord(value.meta) ? value.meta.duration_seconds : null;
  const ocr = isRecord(value.meta) ? value.meta.ocr : null;
  if (ocr != null && !isOcrSummary(ocr)) return null;
  return {
    document: {
      image: document.image, width: document.width, height: document.height,
      detected: document.detected, rotation_degrees: document.rotation_degrees,
      orientation_confidence: document.orientation_confidence,
    },
    fields,
    durationSeconds: isDuration(duration) ? duration : null,
    ocr: ocr ?? null,
  };
}
