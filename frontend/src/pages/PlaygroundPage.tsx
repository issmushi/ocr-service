import { useEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { buildCurl, runTool } from "../api";
import { CameraCapture } from "../components/CameraCapture";
import { PassportFieldsView } from "../components/PassportFieldsView";
import { acceptedFormats, defaultParams, toolById, tools } from "../config";
import type { ParamValue, RequestResult, ToolDefinition, ToolId } from "../types";
import { Icon } from "../components/Icon";

interface Props {
  apiBaseUrl: string;
  initialTool: ToolId;
  onToolChange: (tool: ToolId) => void;
}

type ResultTab = "fields" | "mrz" | "normalization" | "meta" | "json" | "curl";

const fieldLabels: Record<string, string> = {
  passport_type: "Тип паспорта",
  document_type: "Тип документа",
  document_code: "Код документа",
  issuing_country: "Страна выдачи",
  surname: "Фамилия",
  name: "Имя",
  patronymic: "Отчество",
  sex: "Пол",
  date_of_birth: "Дата рождения",
  passport_series: "Серия паспорта",
  passport_number: "Номер паспорта",
  document_series: "Серия документа",
  document_number: "Номер документа",
  nationality: "Гражданство",
  issue_date: "Дата выдачи",
  expiry_date: "Срок действия",
  department_code: "Код подразделения",
  personal_number: "Персональный номер",
  mrz_valid: "MRZ валидна",
  text: "Распознанный текст",
};

const excludedFields = new Set(["mrz", "normalization", "meta"]);

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} КБ`;
  return `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
}

function stringify(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function canPreviewInBrowser(file: File): boolean {
  const extension = file.name.split(".").pop()?.toLowerCase();
  return file.type !== "image/tiff" && extension !== "tif" && extension !== "tiff";
}

function getEnvelopeData(result: RequestResult | null): Record<string, unknown> | null {
  if (!result?.json || typeof result.json !== "object") return null;
  const envelope = result.json as Record<string, unknown>;
  const data = envelope.data;
  return data && typeof data === "object" ? data as Record<string, unknown> : null;
}

function usePreviewUrls(files: File[]): string[] {
  const [preview, setPreview] = useState<{ files: File[]; urls: string[] } | null>(null);
  useEffect(() => {
    const next = files.map((file) => URL.createObjectURL(file));
    setPreview({ files, urls: next });
    return () => next.forEach((url) => URL.revokeObjectURL(url));
  }, [files]);
  return preview?.files === files ? preview.urls : [];
}

function handleTabListKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  const tabs = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]:not(:disabled)'));
  const current = tabs.indexOf(document.activeElement as HTMLButtonElement);
  if (current < 0 || !tabs.length) return;
  event.preventDefault();
  const next = event.key === "Home"
    ? 0
    : event.key === "End"
      ? tabs.length - 1
      : (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
  tabs[next].focus();
  tabs[next].click();
}

export function PlaygroundPage({ apiBaseUrl, initialTool, onToolChange }: Props) {
  const [toolId, setToolId] = useState<ToolId>(initialTool);
  const tool = toolById[toolId];
  const [params, setParams] = useState<Record<string, ParamValue>>(() => defaultParams(tool));
  const [files, setFiles] = useState<File[]>([]);
  const [submittedFiles, setSubmittedFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [result, setResult] = useState<RequestResult | null>(null);
  const [resultCurl, setResultCurl] = useState<string | null>(null);
  const [resultWasNormalized, setResultWasNormalized] = useState(true);
  const [running, setRunning] = useState(false);
  const [activeTab, setActiveTab] = useState<ResultTab>("fields");
  const [copied, setCopied] = useState<string | null>(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraNotice, setCameraNotice] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const previousApiBaseUrl = useRef(apiBaseUrl);
  const inputRef = useRef<HTMLInputElement>(null);
  const replaceFileInput = useRef(false);
  const captureInputRef = useRef<HTMLInputElement>(null);
  const replaceNativeCapture = useRef(false);
  const cameraTriggerRef = useRef<HTMLButtonElement>(null);
  const previews = usePreviewUrls(files);
  const resultImageUrl = result?.imageUrl;

  useEffect(() => {
    setToolId(initialTool);
  }, [initialTool]);

  useEffect(() => {
    controller.current?.abort();
    controller.current = null;
    setParams(defaultParams(tool));
    setFiles([]);
    setSubmittedFiles([]);
    setRunning(false);
    setResult((previous) => {
      if (previous?.imageUrl) URL.revokeObjectURL(previous.imageUrl);
      return null;
    });
    setResultCurl(null);
    setActiveTab(tool.responseKind === "json" ? "fields" : "json");
    setValidationError(null);
    setCameraOpen(false);
    setCameraNotice(null);
  }, [tool]);

  useEffect(() => {
    if (previousApiBaseUrl.current === apiBaseUrl) return;
    previousApiBaseUrl.current = apiBaseUrl;
    controller.current?.abort();
    controller.current = null;
    setRunning(false);
    setResult(null);
    setResultCurl(null);
    setSubmittedFiles([]);
    setValidationError(null);
  }, [apiBaseUrl]);

  useEffect(() => () => {
    controller.current?.abort();
  }, []);

  useEffect(() => () => {
    if (resultImageUrl) URL.revokeObjectURL(resultImageUrl);
  }, [resultImageUrl]);

  const curl = useMemo(() => buildCurl(apiBaseUrl, tool, files, params), [apiBaseUrl, files, params, tool]);
  const data = getEnvelopeData(result);
  const mrz = Array.isArray(data?.mrz) ? data.mrz.filter((line): line is string => typeof line === "string") : [];
  const normalization = data?.normalization && typeof data.normalization === "object" ? data.normalization as Record<string, unknown> : null;
  const meta = data?.meta && typeof data.meta === "object" ? data.meta as Record<string, unknown> : null;
  const fieldEntries = data ? Object.entries(data).filter(([key]) => !excludedFields.has(key)) : [];

  const chooseTool = (next: ToolId) => {
    setToolId(next);
    onToolChange(next);
  };

  const invalidateResult = () => {
    controller.current?.abort();
    controller.current = null;
    setRunning(false);
    setResult(null);
    setResultCurl(null);
    setSubmittedFiles([]);
  };

  const updateParam = (key: string, value: ParamValue) => {
    invalidateResult();
    setParams((current) => ({ ...current, [key]: value }));
  };

  const acceptFiles = (incoming: File[], replace = false) => {
    if (!incoming.length) return;
    const maxFiles = tool.maxFiles ?? 1;
    const allowedExtensions = new Set(["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"]);
    const invalid = incoming.find((file) => !allowedExtensions.has(file.name.split(".").pop()?.toLowerCase() ?? ""));
    if (invalid) {
      setValidationError(`Формат файла «${invalid.name}» не поддерживается.`);
      return;
    }
    invalidateResult();
    setValidationError(null);
    setFiles((current) => tool.multiple && !replace ? [...current, ...incoming].slice(0, maxFiles) : incoming.slice(0, maxFiles));
  };

  const closeCamera = () => {
    setCameraOpen(false);
    window.requestAnimationFrame(() => cameraTriggerRef.current?.focus());
  };

  const openFiles = (replace = false) => {
    replaceFileInput.current = replace;
    inputRef.current?.click();
  };

  const openSystemCamera = () => {
    replaceNativeCapture.current = files.length >= (tool.maxFiles ?? 1);
    setCameraNotice(toolId === "best-frame"
        ? "Автоматическая серия доступна через HTTPS или localhost. Здесь можно снять три фото по отдельности через системную камеру или выбрать готовые файлы, затем нажать «Выбрать лучший кадр»."
        : "Живая камера доступна через HTTPS или localhost. Используйте системную камеру или выберите файл.");
    captureInputRef.current?.click();
  };

  const openCamera = () => {
    setValidationError(null);
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      openSystemCamera();
      return;
    }
    setCameraNotice(null);
    setCameraOpen(true);
  };

  const acceptCameraFiles = (capturedFiles: File[]) => {
    closeCamera();
    if (toolId === "best-frame") {
      setFiles(capturedFiles);
      void submit(capturedFiles);
    } else {
      acceptFiles(capturedFiles);
    }
  };

  const removeFile = (index: number) => {
    invalidateResult();
    setFiles((current) => current.filter((_, currentIndex) => currentIndex !== index));
  };

  const submit = async (requestFiles: File[] = files) => {
    if (!requestFiles.length) {
      setValidationError("Добавьте изображение для запуска.");
      return;
    }
    controller.current?.abort();
    const nextController = new AbortController();
    const submittedCurl = buildCurl(apiBaseUrl, tool, requestFiles, params);
    const submittedNormalize = params.normalize !== false;
    controller.current = nextController;
    setRunning(true);
    setSubmittedFiles([...requestFiles]);
    setValidationError(null);
    setResultCurl(null);
    setResult((previous) => {
      if (previous?.imageUrl) URL.revokeObjectURL(previous.imageUrl);
      return null;
    });
    try {
      const next = await runTool(apiBaseUrl, tool, requestFiles, params, nextController.signal);
      if (controller.current !== nextController) {
        if (next.imageUrl) URL.revokeObjectURL(next.imageUrl);
        return;
      }
      setResult(next);
      setResultCurl(submittedCurl);
      setResultWasNormalized(submittedNormalize);
      setActiveTab(tool.responseKind === "json" ? "fields" : "json");
    } catch (error) {
      if (controller.current !== nextController) return;
      if (error instanceof DOMException && error.name === "AbortError") return;
      setValidationError(error instanceof Error ? error.message : "Не удалось выполнить запрос");
    } finally {
      if (controller.current === nextController) {
        controller.current = null;
        setRunning(false);
      }
    }
  };

  const stop = () => {
    controller.current?.abort();
    controller.current = null;
    setRunning(false);
  };

  const copy = async (key: string, value: string) => {
    await navigator.clipboard.writeText(value);
    setCopied(key);
    window.setTimeout(() => setCopied(null), 1400);
  };

  return (
    <div className="page page--playground">
      <header className="page-heading playground-heading">
        <div><span className="eyebrow">Проверка API</span><h1>Тестирование</h1><p>Выберите эндпоинт, добавьте фото и задайте параметры. Здесь же появятся ответ и время запроса.</p></div>
      </header>

      <div className="playground-layout">
        <aside className="tool-picker">
          {(["Распознавание", "Обрезка и качество"] as const).map((group) => (
            <div key={group}>
              <span className="nav-kicker">{group}</span>
              {tools.filter((item) => item.group === group).map((item) => (
                <button className={item.id === toolId ? "active" : ""} key={item.id} onClick={() => chooseTool(item.id)} type="button">
                  <span><Icon name={item.id.includes("crop") ? "crop" : item.id === "best-frame" ? "frame" : item.id === "passport-page" ? "passport" : item.id === "full-ocr" ? "file" : "scan"} /></span>
                  <div><strong>{item.title}</strong><small>{item.shortTitle}</small></div>
                  <Icon name="chevron" size={15} />
                </button>
              ))}
            </div>
          ))}
        </aside>

        <section className="request-panel panel">
          <div className="panel__head">
            <div><span className="method">POST</span><code>{tool.endpoint}</code></div>
            <span className="response-type">{tool.responseKind === "json" ? "JSON" : "IMAGE"}</span>
          </div>
          <div className="request-intro"><h2>{tool.title}</h2><p>{tool.description}</p></div>

          <div
            className={`dropzone ${dragging ? "dropzone--active" : ""} ${files.length ? "dropzone--filled" : ""}`}
            onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={(event) => { event.preventDefault(); setDragging(false); }}
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => { event.preventDefault(); setDragging(false); acceptFiles(Array.from(event.dataTransfer.files)); }}
          >
            <input
              accept={acceptedFormats}
              hidden
              multiple={tool.multiple}
              onChange={(event) => { acceptFiles(Array.from(event.target.files ?? []), replaceFileInput.current); event.target.value = ""; replaceFileInput.current = false; }}
              ref={inputRef}
              type="file"
            />
            <input
              accept="image/jpeg,image/png,image/webp"
              capture="environment"
              hidden
              onChange={(event) => { acceptFiles(Array.from(event.target.files ?? []), replaceNativeCapture.current); event.target.value = ""; replaceNativeCapture.current = false; }}
              ref={captureInputRef}
              type="file"
            />
            {!files.length ? (
              <><span className="dropzone__icon"><Icon name="camera" size={24} /></span><strong>{tool.multiple ? "Добавьте до 3 фото одного документа" : "Добавьте фото документа"}</strong><p>{tool.multiple ? "Можно выбрать файлы или снять серию. Камера сделает 3 фото и сразу отправит их на сравнение." : "Перетащите файл сюда, выберите его с диска или снимите документ камерой."}</p><div className="dropzone__actions"><button onClick={() => openFiles()} type="button"><Icon name="upload" size={16} />Выбрать файл</button><button onClick={openCamera} ref={cameraTriggerRef} type="button"><Icon name="camera" size={16} />{tool.multiple ? "Снять серию" : "Снять документ"}</button></div><small>JPG, PNG, BMP, WebP, TIFF · лимиты — в настройках API</small></>
            ) : (
              <div className={`file-preview-list ${tool.multiple ? "file-preview-list--multiple" : ""}`}>
                {files.map((file, index) => (
                  <div className="file-preview" key={`${file.name}-${file.lastModified}-${index}`} onClick={(event) => event.stopPropagation()}>
                    <div className="file-preview__image">{previews[index] && canPreviewInBrowser(file) ? <img alt={`Предпросмотр ${file.name}`} src={previews[index]} /> : <div className="file-preview__placeholder">TIFF</div>}{tool.multiple && <span>{index + 1}</span>}</div>
                    <div className="file-preview__details"><strong title={file.name}>{file.name}</strong><small>{formatBytes(file.size)}</small></div>
                    <button aria-label={`Удалить ${file.name}`} onClick={() => removeFile(index)} type="button"><Icon name="x" size={16} /></button>
                  </div>
                ))}
                <div className="file-preview-actions">
                  {(!tool.multiple || files.length < (tool.maxFiles ?? 3)) && <button className="add-frame" onClick={() => openFiles()} type="button"><Icon name="upload" size={16} />{tool.multiple ? "Добавить файл" : "Заменить файлом"}</button>}
                  <button className="add-frame" onClick={openCamera} ref={cameraTriggerRef} type="button"><Icon name="camera" size={16} />{tool.multiple ? "Снять новую серию" : "Переснять"}</button>
                </div>
              </div>
            )}
          </div>

          {cameraNotice && <p className="camera-notice" role="status">{cameraNotice}</p>}

          {tool.params.length > 0 && (
            <div className="params-section">
              <div className="subheading"><span>Параметры</span><button onClick={() => { invalidateResult(); setParams(defaultParams(tool)); }} type="button">Сбросить</button></div>
              <div className="params-list">
                {tool.params.map((param) => param.kind === "select" ? (
                  <label className="param-row param-row--select" key={param.key}>
                    <span><strong>{param.label}</strong><small>{param.description}</small></span>
                    <select value={String(params[param.key])} onChange={(event) => updateParam(param.key, event.target.value)}>
                      {param.options.map((option) => <option key={option.value} value={option.value}>{option.label}{option.note ? ` — ${option.note}` : ""}</option>)}
                    </select>
                  </label>
                ) : (
                  <label className="param-row" key={param.key}>
                    <span><strong>{param.label}</strong><small>{param.description}</small></span>
                    <input checked={Boolean(params[param.key])} onChange={(event) => updateParam(param.key, event.target.checked)} type="checkbox" />
                  </label>
                ))}
              </div>
            </div>
          )}

          {validationError && <div className="inline-error" role="alert"><Icon name="info" size={18} /><span>{validationError}</span></div>}
          <div className="request-actions">
            <span>{tool.outputHint}</span>
            {running ? (
              <button className="button button--danger" onClick={stop} type="button"><Icon name="stop" size={17} />Прервать ожидание</button>
            ) : (
              <button className="button button--primary" disabled={!files.length} onClick={() => void submit()} type="button"><Icon name="play" size={17} />{tool.multiple ? "Выбрать лучший кадр" : "Запустить"}</button>
            )}
          </div>
        </section>

        <section aria-busy={running} aria-live="polite" className={`result-panel panel ${toolId === "best-frame" ? "result-panel--best-frame" : ""}`}>
          <div className="panel__head result-panel__head">
            <div><span>Результат</span>{result && <span className={`result-status ${result.ok ? "result-status--ok" : "result-status--error"}`}>{result.status} {result.ok ? "OK" : result.statusText}</span>}</div>
            {result && <span className="result-duration"><Icon name="clock" size={14} />{(result.durationMs / 1000).toFixed(2)} с</span>}
          </div>

          {running ? (
            <div className="processing-state" role="status"><div className="scanner-animation"><span /></div><strong>{tool.multiple ? "Сравниваем кадры" : "Ждём ответ API"}</strong><p>{tool.multiple ? "Сервер оценивает резкость, освещение, блики и границы документа." : "Первый запрос может быть дольше, пока модель загружается."}</p><div className="processing-steps"><span className="done">Загрузка</span><span className="active">{tool.multiple ? "Оценка качества" : "Обработка"}</span><span>Ответ</span></div></div>
          ) : !result ? (
            <div className="empty-result"><span><Icon name="code" size={28} /></span><strong>Ответ API</strong><p>Добавьте фото и отправьте запрос. Здесь появятся результат и время запроса.</p></div>
          ) : !result.ok ? (
            <div className="error-result" role="alert">
              <span><Icon name="info" size={25} /></span><div><small>Ошибка запроса</small><h3>{result.errorMessage}</h3><p>HTTP {result.status} · {result.contentType || "неизвестный формат"}</p></div>
              <details><summary>Ответ сервера</summary><pre>{stringify(result.json)}</pre></details>
            </div>
          ) : tool.responseKind === "image" && result.imageUrl ? (
            <ImageResult result={result} tool={tool} onCopy={copy} copied={copied} curl={resultCurl ?? curl} />
          ) : (
            <div className="structured-result">
              <div className="result-tabs" onKeyDown={handleTabListKeyDown} role="tablist">
                <ResultTabButton active={activeTab === "fields"} id="fields" label={toolId === "crop-passport-fields" ? "Вырезки" : "Поля"} onClick={() => setActiveTab("fields")} />
                {toolId !== "crop-passport-fields" && <ResultTabButton active={activeTab === "mrz"} count={mrz.length || undefined} disabled={!mrz.length} id="mrz" label="MRZ" onClick={() => setActiveTab("mrz")} />}
                {toolId !== "crop-passport-fields" && <ResultTabButton active={activeTab === "normalization"} count={typeof normalization?.corrections_count === "number" ? normalization.corrections_count : undefined} disabled={!normalization} id="normalization" label="Нормализация" onClick={() => setActiveTab("normalization")} />}
                <ResultTabButton active={activeTab === "meta"} disabled={!meta} id="meta" label="Meta" onClick={() => setActiveTab("meta")} />
                <ResultTabButton active={activeTab === "json"} id="json" label="JSON" onClick={() => setActiveTab("json")} />
                <ResultTabButton active={activeTab === "curl"} id="curl" label="cURL" onClick={() => setActiveTab("curl")} />
              </div>
              <div aria-labelledby={`result-tab-${activeTab}`} className="result-body" id={`result-panel-${activeTab}`} role="tabpanel" tabIndex={0}>
                {activeTab === "fields" && (toolId === "crop-passport-fields" ? <PassportFieldsView data={data} /> : <FieldsView entries={fieldEntries} />)}
                {activeTab === "mrz" && <MrzView lines={mrz} copied={copied} normalized={resultWasNormalized} onCopy={copy} />}
                {activeTab === "normalization" && <NormalizationView enabled={resultWasNormalized} value={normalization} />}
                {activeTab === "meta" && <MetaView meta={meta} clientDuration={result.durationMs} />}
                {activeTab === "json" && <CodeView copied={copied === "json"} language="JSON" onCopy={() => void copy("json", stringify(result.json))} value={stringify(result.json)} />}
                {activeTab === "curl" && <CodeView copied={copied === "curl"} language="Shell" onCopy={() => void copy("curl", resultCurl ?? curl)} value={resultCurl ?? curl} />}
              </div>
            </div>
          )}
          {toolId === "best-frame" && submittedFiles.length > 0 && (
            <SubmittedFrames
              files={submittedFiles}
              selectedIndex={result?.ok ? result.headers["x-selected-index"] : undefined}
            />
          )}
        </section>
      </div>
      {cameraOpen && (
        <CameraCapture
          frameCount={toolId === "best-frame" ? 3 : 1}
          onCapture={acceptCameraFiles}
          onClose={closeCamera}
          onFallback={() => toolId === "best-frame" ? openFiles(true) : openSystemCamera()}
        />
      )}
    </div>
  );
}

function SubmittedFrames({ files, selectedIndex }: { files: File[]; selectedIndex?: string }) {
  const urls = usePreviewUrls(files);
  const selected = selectedIndex === undefined ? -1 : Number(selectedIndex);
  return (
    <section aria-label="Отправленные кадры" className="submitted-frames">
      <div className="subheading"><span>Отправленные кадры · {files.length}</span></div>
      <p>Исходные снимки из браузера. Сервер возвращает только лучший кадр.</p>
      <div className="submitted-frames__grid">
        {files.map((file, index) => (
          <figure className={index === selected ? "submitted-frame submitted-frame--selected" : "submitted-frame"} key={`${file.name}-${index}`}>
            <a aria-label={`Открыть отправленный кадр ${index + 1}`} href={urls[index]} rel="noreferrer" target="_blank">
              {urls[index] && canPreviewInBrowser(file)
                ? <img alt={`Отправленный кадр ${index + 1}`} src={urls[index]} />
                : <span className="submitted-frame__placeholder">TIFF · открыть файл</span>}
            </a>
            <figcaption><strong>Кадр {index + 1}</strong>{index === selected && <span><Icon name="check" size={12} />Лучший</span>}<small>{formatBytes(file.size)}</small></figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}

function ResultTabButton({ active, count, disabled, id, label, onClick }: { active: boolean; count?: number; disabled?: boolean; id: string; label: string; onClick: () => void }) {
  return <button aria-controls={`result-panel-${id}`} aria-selected={active} className={active ? "active" : ""} disabled={disabled} id={`result-tab-${id}`} onClick={onClick} role="tab" tabIndex={active ? 0 : -1} type="button">{label}{count !== undefined && <span>{count}</span>}</button>;
}

function FieldsView({ entries }: { entries: [string, unknown][] }) {
  if (!entries.length) return <div className="tab-empty">В ответе нет полей документа.</div>;
  return <div className="fields-grid">{entries.map(([key, value]) => (
    key === "text" ? (
      <div className="field-item field-item--text" key={key}>
        <span>{fieldLabels[key]}</span>
        <pre>{String(value ?? "")}</pre>
      </div>
    ) :
    <div className={key === "mrz_valid" ? "field-item field-item--highlight" : "field-item"} key={key}>
      <span>{fieldLabels[key] ?? key.replaceAll("_", " ")}</span>
      <strong className={value === null || value === "" ? "muted" : ""}>{value === null || value === "" ? "Не найдено" : typeof value === "boolean" ? value ? "Да" : "Нет" : String(value)}</strong>
    </div>
  ))}</div>;
}

function MrzView({ lines, copied, normalized, onCopy }: { lines: string[]; copied: string | null; normalized: boolean; onCopy: (key: string, value: string) => Promise<void> }) {
  const combined = lines.join("\n");
  return <div className="mrz-view"><div className="mrz-view__head"><span>{lines.length} строки</span><button onClick={() => void onCopy("mrz", combined)} type="button"><Icon name={copied === "mrz" ? "check" : "copy"} size={15} />{copied === "mrz" ? "Скопировано" : "Копировать"}</button></div>{lines.map((line, index) => <div className="mrz-line" key={`${line}-${index}`}><span>0{index + 1}</span><code>{line}</code><small>{line.length}</small></div>)}<p><Icon name="info" size={15} /> Число справа — длина строки {normalized ? "после нормализации" : "после выбора строк OCR"}.</p></div>;
}

function NormalizationView({ enabled, value }: { enabled: boolean; value: Record<string, unknown> | null }) {
  if (!value) return <div className="tab-empty">В ответе нет журнала нормализации.</div>;
  const corrections = Array.isArray(value.corrections) ? value.corrections.filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object")) : [];
  const issues = Array.isArray(value.issues) ? value.issues.filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object")) : [];
  return <div className="normalization-view">
    <div className="normalization-stats"><div><strong>{String(value.corrections_count ?? 0)}</strong><span>всего</span></div><div><strong>{String(value.replacements_count ?? 0)}</strong><span>замен</span></div><div><strong>{String(value.insertions_count ?? 0)}</strong><span>вставок</span></div><div><strong>{issues.length}</strong><span>замечаний</span></div></div>
    {corrections.length ? <div className="correction-list">{corrections.map((correction, index) => <div className="correction" key={index}><span className="correction__line">L{String(correction.line ?? "–")}</span><div><div><strong>{String(correction.field ?? "Поле MRZ")}</strong><span>{String(correction.operation ?? "изменение")}</span></div><code><del>{String(correction.source ?? "") || "∅"}</del><Icon name="arrow" size={14} /><ins>{String(correction.replacement ?? "") || "∅"}</ins></code><p>{String(correction.reason ?? "")}</p></div><small>поз. {String(correction.position ?? "–")}</small></div>)}</div> : <div className="normalization-clean"><span><Icon name={enabled ? "check" : "info"} /></span><div><strong>{enabled ? "Замен и вставок нет" : "Нормализация отключена"}</strong><p>{enabled ? "Нормализатор не изменил символы. Если есть замечания, они показаны ниже." : "Правила исправления MRZ не применялись."}</p></div></div>}
    {issues.length > 0 && <div className="issue-list"><strong>Замечания</strong>{issues.map((issue, index) => <div key={index}><Icon name="info" size={16} /><span>{String(issue.message ?? issue.code ?? "Неизвестное замечание")}</span></div>)}</div>}
  </div>;
}

function MetaView({ meta, clientDuration }: { meta: Record<string, unknown> | null; clientDuration: number }) {
  return <div className="meta-view"><div className="meta-card"><span><Icon name="settings" /></span><small>Движок</small><strong>{String(meta?.engine ?? "—")}</strong></div><div className="meta-card"><span><Icon name="layers" /></span><small>Модель</small><strong>{String(meta?.model ?? "—")}</strong></div><div className="meta-card"><span><Icon name="activity" /></span><small>OCR на сервере</small><strong>{typeof meta?.duration_seconds === "number" ? `${meta.duration_seconds.toFixed(3)} с` : "—"}</strong></div><div className="meta-card"><span><Icon name="clock" /></span><small>Время запроса</small><strong>{(clientDuration / 1000).toFixed(3)} с</strong></div></div>;
}

function CodeView({ copied, language, onCopy, value }: { copied: boolean; language: string; onCopy: () => void; value: string }) {
  return <div className="code-result"><div><span>{language}</span><button onClick={onCopy} type="button"><Icon name={copied ? "check" : "copy"} size={15} />{copied ? "Скопировано" : "Копировать"}</button></div><pre><code>{value}</code></pre></div>;
}

function ImageResult({ copied, curl, onCopy, result, tool }: { copied: string | null; curl: string; onCopy: (key: string, value: string) => Promise<void>; result: RequestResult; tool: ToolDefinition }) {
  const [tab, setTab] = useState<"preview" | "headers" | "curl">("preview");
  const selected = result.headers["x-selected-index"];
  const score = result.headers["x-quality-score"];
  const previewSupported = result.contentType.split(";")[0].toLowerCase() !== "image/tiff";
  const download = () => {
    if (!result.imageUrl) return;
    const extensionByContentType: Record<string, string> = {
      "image/jpeg": "jpg",
      "image/png": "png",
      "image/webp": "webp",
      "image/bmp": "bmp",
      "image/tiff": "tiff",
    };
    const extension = extensionByContentType[result.contentType.split(";")[0].toLowerCase()] ?? "img";
    const link = document.createElement("a");
    link.href = result.imageUrl;
    link.download = `${tool.id}.${extension}`;
    link.click();
  };
  return <div className="structured-result image-result"><div className="result-tabs" onKeyDown={handleTabListKeyDown} role="tablist"><ResultTabButton active={tab === "preview"} id="image-preview" label="Изображение" onClick={() => setTab("preview")} /><ResultTabButton active={tab === "headers"} id="image-headers" label="Заголовки" onClick={() => setTab("headers")} /><ResultTabButton active={tab === "curl"} id="image-curl" label="cURL" onClick={() => setTab("curl")} /></div><div aria-labelledby={`result-tab-image-${tab}`} className="result-body" id={`result-panel-image-${tab}`} role="tabpanel" tabIndex={0}>
    {tab === "preview" && <div className="output-image"><div className="output-image__meta"><div><span className="result-status result-status--ok"><Icon name="check" size={13} />Готово</span>{selected !== undefined && <span>Выбран кадр №{Number(selected) + 1}</span>}{score !== undefined && <span>Качество {Number(score).toFixed(3)}</span>}</div><button onClick={download} type="button"><Icon name="download" size={16} />Скачать</button></div>{previewSupported ? <img alt="Результат обработки" src={result.imageUrl} /> : <div className="output-image__unsupported"><Icon name="file" size={26} /><strong>Результат в TIFF</strong><span>Предпросмотр TIFF здесь недоступен. Скачайте файл, чтобы открыть его.</span></div>}<small>{result.contentType} · {result.imageBlob ? formatBytes(result.imageBlob.size) : ""}</small></div>}
    {tab === "headers" && <div className="headers-list">{Object.entries(result.headers).map(([key, value]) => <div key={key}><code>{key}</code><span>{value}</span></div>)}</div>}
    {tab === "curl" && <CodeView copied={copied === "curl"} language="Shell" onCopy={() => void onCopy("curl", curl)} value={curl} />}
  </div></div>;
}
