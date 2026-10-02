export type Section = "overview" | "docs" | "playground";

export type ToolId =
  | "mrz"
  | "passport-internal"
  | "full-ocr"
  | "crop-document"
  | "crop-mrz"
  | "crop-passport-fields"
  | "best-frame";

export type ResponseKind = "json" | "image";

export type ParamValue = string | boolean;

export interface SelectOption {
  value: string;
  label: string;
  note?: string;
}

export interface SelectParam {
  kind: "select";
  key: string;
  label: string;
  description: string;
  defaultValue: string;
  options: SelectOption[];
}

export interface ToggleParam {
  kind: "toggle";
  key: string;
  label: string;
  description: string;
  defaultValue: boolean;
}

export type ParamDefinition = SelectParam | ToggleParam;

export interface ToolDefinition {
  id: ToolId;
  group: "MRZ" | "Распознавание" | "Обрезка и качество";
  title: string;
  shortTitle: string;
  description: string;
  endpoint: string;
  responseKind: ResponseKind;
  multiple?: boolean;
  maxFiles?: number;
  params: ParamDefinition[];
  outputHint: string;
}

export interface ApiErrorBody {
  code?: string;
  message?: string;
}

export interface ApiEnvelope<T = Record<string, unknown>> {
  success: boolean;
  data: T | null;
  error: ApiErrorBody | null;
}

export interface RequestResult {
  ok: boolean;
  status: number;
  statusText: string;
  durationMs: number;
  contentType: string;
  json?: unknown;
  imageUrl?: string;
  imageBlob?: Blob;
  headers: Record<string, string>;
  errorMessage?: string;
}

export interface HealthState {
  status: "idle" | "checking" | "online" | "degraded" | "offline";
  version?: string;
  latencyMs?: number;
  message?: string;
}
