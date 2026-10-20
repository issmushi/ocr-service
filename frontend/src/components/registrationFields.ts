import {
  isCropImage, isFieldOcr, isProbability, isRecord, parsePassportFieldCrops,
  type PassportFieldCrop, type PassportFieldCrops, type Quad,
} from "./passportFields";

function isQuad(value: unknown): value is Quad {
  if (!Array.isArray(value) || value.length !== 4
    || !value.every((point) => Array.isArray(point) && point.length === 2
      && point.every((coordinate) => typeof coordinate === "number" && Number.isFinite(coordinate)))) return false;
  const area = value.reduce((sum, point, index) => {
    const next = value[(index + 1) % 4];
    return sum + point[0] * next[1] - next[0] * point[1];
  }, 0);
  // Padding may extend beyond the document. Preserve those coordinates.
  return Number.isFinite(area) && Math.abs(area) > 0;
}

function enclosingBox(quad: Quad): [number, number, number, number] {
  const x = quad.map((point) => point[0]);
  const y = quad.map((point) => point[1]);
  return [Math.min(...x), Math.min(...y), Math.max(...x), Math.max(...y)];
}

export function parseRegistrationFieldCrops(value: unknown): PassportFieldCrops | null {
  if (!isRecord(value) || !Array.isArray(value.fields)
    || typeof value.document_type !== "string") return null;
  const common = parsePassportFieldCrops({ ...value, fields: [] });
  if (!common) return null;

  const fields: PassportFieldCrop[] = [];
  for (const field of value.fields) {
    if (!isRecord(field) || !isCropImage(field, true)
      || (field.kind !== "address_line" && field.kind !== "registration_stamp")
      || typeof field.name !== "string" || !field.name.trim()
      || typeof field.label !== "string" || !field.label.trim()
      || !isQuad(field.quad) || !isQuad(field.crop_quad)
      || !isProbability(field.confidence)
      || (field.ocr != null && !isFieldOcr(field.ocr))) return null;
    fields.push({
      name: field.name, label: field.label, image: field.image,
      width: field.width, height: field.height, rotation_degrees: field.rotation_degrees,
      kind: field.kind,
      quad: field.quad, crop_quad: field.crop_quad,
      bbox: enclosingBox(field.crop_quad), detection_bbox: enclosingBox(field.quad),
      confidence: field.confidence, ocr: field.ocr ?? null,
    });
  }
  return { ...common, fields };
}
