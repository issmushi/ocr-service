import type { ParamValue, RequestResult, ToolDefinition } from "./types";

export function normalizeBaseUrl(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

export function resolveApiUrl(baseUrl: string, endpoint: string): string {
  return `${normalizeBaseUrl(baseUrl)}${endpoint}`;
}

function shellQuote(value: string): string {
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}

export function buildCurl(
  baseUrl: string,
  tool: ToolDefinition,
  files: File[],
  params: Record<string, ParamValue>,
): string {
  const fallbackOrigin = globalThis.location?.origin ?? "http://127.0.0.1:8000";
  const url = resolveApiUrl(baseUrl || fallbackOrigin, tool.endpoint);
  const lines = [`curl -X POST ${shellQuote(url)}`];
  const sourceFiles = files.length ? files : [new File([], "document.jpg")];
  sourceFiles.forEach((file, index) => {
    const displayName = files.length ? file.name : "document.jpg";
    const prefix = tool.multiple ? `frame-${index + 1}/` : "";
    lines.push(`  -F ${shellQuote(`image=@/path/to/${prefix}${displayName}`)}`);
  });
  Object.entries(params).forEach(([key, value]) => {
    lines.push(`  -F ${shellQuote(`${key}=${String(value)}`)}`);
  });
  if (tool.responseKind === "image") {
    lines.push(`  --output ${shellQuote(`${tool.id}.jpg`)}`);
  }
  return lines.map((line, index) => `${index ? "\\\n" : ""}${line}`).join("");
}

function extractError(payload: unknown, fallback: string): string {
  if (payload && typeof payload === "object") {
    const object = payload as Record<string, unknown>;
    const error = object.error;
    if (error && typeof error === "object") {
      const message = (error as Record<string, unknown>).message;
      if (typeof message === "string") return message;
    }
    if (typeof object.detail === "string") return object.detail;
    if (Array.isArray(object.detail)) {
      return object.detail
        .map((item) => {
          if (item && typeof item === "object" && "msg" in item) return String(item.msg);
          return String(item);
        })
        .join("; ");
    }
  }
  return fallback;
}

export async function runTool(
  baseUrl: string,
  tool: ToolDefinition,
  files: File[],
  params: Record<string, ParamValue>,
  signal: AbortSignal,
): Promise<RequestResult> {
  const form = new FormData();
  files.forEach((file) => form.append("image", file, file.name));
  Object.entries(params).forEach(([key, value]) => form.append(key, String(value)));

  const started = performance.now();
  let response: Response;
  try {
    response = await fetch(resolveApiUrl(baseUrl, tool.endpoint), {
      method: "POST",
      body: form,
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    const message = error instanceof Error ? error.message : "Сетевая ошибка";
    throw new Error(`Не удалось подключиться к API: ${message}`);
  }

  const durationMs = performance.now() - started;
  const contentType = response.headers.get("content-type") || "";
  const headers = Object.fromEntries(response.headers.entries());

  if (tool.responseKind === "image" && response.ok && contentType.startsWith("image/")) {
    const imageBlob = await response.blob();
    return {
      ok: true,
      status: response.status,
      statusText: response.statusText,
      durationMs,
      contentType,
      imageBlob,
      imageUrl: URL.createObjectURL(imageBlob),
      headers,
    };
  }

  let payload: unknown;
  const body = await response.text();
  try {
    payload = body ? JSON.parse(body) : null;
  } catch {
    payload = { detail: body || "Сервер вернул ответ неизвестного формата" };
  }

  return {
    ok: response.ok && !(payload && typeof payload === "object" && "success" in payload && !(payload as { success: boolean }).success),
    status: response.status,
    statusText: response.statusText,
    durationMs,
    contentType,
    json: payload,
    headers,
    errorMessage: response.ok ? extractError(payload, "Распознавание завершилось с ошибкой") : extractError(payload, `HTTP ${response.status}`),
  };
}
