import assert from "node:assert/strict";
import { File as NodeFile } from "node:buffer";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

const require = createRequire(import.meta.url);
globalThis.File ??= NodeFile;

async function sourceUrl(path, imports = {}) {
  const source = await readFile(new URL(path, import.meta.url), "utf8");
  let { outputText } = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022, jsx: ts.JsxEmit.ReactJSX },
  });
  for (const [specifier, url] of Object.entries(imports)) outputText = outputText.replaceAll(`from "${specifier}"`, `from "${url}"`);
  return `data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`;
}

const parserUrl = await sourceUrl("../src/components/passportFields.ts");
const { parsePassportFieldCrops } = await import(parserUrl);
const configUrl = await sourceUrl("../src/config.ts");
const { toolById, defaultParams } = await import(configUrl);
const { PassportFieldsView } = await import(await sourceUrl("../src/components/PassportFieldsView.tsx", {
  "./passportFields": parserUrl,
  "../config": configUrl,
  react: pathToFileURL(require.resolve("react")).href,
  "react/jsx-runtime": pathToFileURL(require.resolve("react/jsx-runtime")).href,
}));
const { runTool, buildCurl } = await import(await sourceUrl("../src/api.ts"));

function sample() {
  const image = "data:image/jpeg;base64,/9j/2Q==";
  return {
    document: { image, width: 600, height: 400, detected: true, rotation_degrees: 0, orientation_confidence: 0.91 },
    fields: [
      { name: "surname", label: "Фамилия", bbox: [96, 46, 304, 84], detection_bbox: [100, 50, 300, 80], confidence: 0.94, image, width: 208, height: 38, rotation_degrees: 0 },
      { name: "number", label: "Серия и номер", bbox: [576, 0, 600, 300], detection_bbox: [580, 0, 600, 296], confidence: 0.86, image, width: 300, height: 24, rotation_degrees: 90 },
    ],
    meta: { engine: "russian-docs-ocr", model: "text-fields-detector", duration_seconds: 0.251, timings: { fields: 0.2 } },
  };
}

function ocr(overrides = {}) {
  return {
    model: "paddle", engine: "paddleocr", device: "cpu", status: "success",
    text: "ПРИМЕР\nСТРОКА", confidence: 0.87, duration_seconds: 0.123, error: null,
    ...overrides,
  };
}

test("field result renders numbered detector boxes and padded, rotated crops in document coordinates", () => {
  const data = sample();
  const parsed = parsePassportFieldCrops(data);
  assert.deepEqual(parsed.fields[1].bbox, [576, 0, 600, 300]);
  assert.deepEqual([parsed.fields[1].width, parsed.fields[1].height], [300, 24]);
  const markup = renderToStaticMarkup(createElement(PassportFieldsView, { data }));
  assert.match(markup, /viewBox="0 0 600 400"/);
  assert.match(markup, /class="passport-fields__box" height="30"[^>]*width="200" x="100" y="50"/);
  assert.equal((markup.match(/class="passport-field"/g) ?? []).length, 2);
  assert.match(markup, /2\. Серия и номер/);
  assert.match(markup, /90° против часовой стрелки/);
  assert.match(markup, /208 × 38 px/);
  assert.match(markup, /94\.0%/);
  assert.match(markup, /Обработка: 0\.25 с/);
  assert.equal(parsed.ocr, null);
  assert.equal(parsed.fields[0].ocr, null);
  assert.match(markup, /Распознавание текста выключено/);
});

test("raw multiline OCR is shown beside each crop with separate confidence, device and timings", () => {
  const data = sample();
  data.fields[0].ocr = ocr({ text: "ПРИМЕР\n<script>СТРОКА</script>" });
  data.fields[1].ocr = ocr({ text: null, confidence: null });
  data.meta.ocr = { model: "paddle", engine: "paddleocr", duration_seconds: 0.5, processed_fields: 2, failed_fields: 0 };
  const markup = renderToStaticMarkup(createElement(PassportFieldsView, { data }));
  assert.match(markup, /ПРИМЕР\n&lt;script&gt;СТРОКА&lt;\/script&gt;/);
  assert.doesNotMatch(markup, /<script>/);
  assert.match(markup, /Текст не найден/);
  assert.match(markup, /Уверенность детектора<\/dt><dd>94\.0%/);
  assert.match(markup, /Уверенность OCR<\/dt><dd>87\.0%/);
  assert.match(markup, /Устройство OCR<\/dt><dd>CPU/);
  assert.match(markup, /Время OCR<\/dt><dd>0\.12 с/);
  assert.match(markup, /OCR: Paddle eslav v5 · CPU · 0\.50 с/);
  assert.match(markup, /Обработано полей: 2 · Ошибок: 0/);
  assert.match(markup, /Каждое текстовое поле распознаётся отдельно, без нормализации/);
});

test("CPU alternatives show the selected model label in each crop and the OCR summary", () => {
  for (const [model, label, engine] of [
    ["paddle-cyrillic", "Paddle Cyrillic v5 · CPU", "paddleocr"],
    ["russian-ocr-accurate", "Russian OCR accurate · CPU", "russian-docs-ocr"],
  ]) {
    const data = sample();
    data.fields[0].ocr = ocr({ model, engine });
    data.meta.ocr = { model, engine, duration_seconds: 0.5, processed_fields: 1, failed_fields: 0 };
    const parsed = parsePassportFieldCrops(data);
    assert.equal(parsed.fields[0].ocr.model, model);
    assert.equal(parsed.ocr.model, model);
    const markup = renderToStaticMarkup(createElement(PassportFieldsView, { data }));
    assert.ok(markup.includes(`<dt>Модель OCR</dt><dd>${label}</dd>`));
    assert.ok(markup.includes(`OCR: ${label} · 0.50 с`));
    assert.match(markup, /Устройство OCR<\/dt><dd>CPU/);
    assert.match(markup, /ПРИМЕР\nСТРОКА/);
  }
});

test("JPEG document previews and lossless PNG field crops render together", () => {
  const data = sample();
  const png = "data:image/png;base64,iVBORw0KGgo=";
  data.fields = data.fields.map((field) => ({ ...field, image: png }));
  const parsed = parsePassportFieldCrops(data);
  assert.equal(parsed.document.image, data.document.image);
  assert.deepEqual(parsed.fields.map((field) => field.image), [png, png]);
  const markup = renderToStaticMarkup(createElement(PassportFieldsView, { data }));
  assert.match(markup, /<image[^>]*href="data:image\/jpeg;base64,/);
  assert.equal((markup.match(/<img[^>]*src="data:image\/png;base64,/g) ?? []).length, 2);
  assert.doesNotMatch(markup, /неожиданный формат/);
});

test("one field error or skipped non-text region does not hide other crops or recognized text", () => {
  const data = sample();
  data.fields[0].ocr = ocr();
  data.fields[1].ocr = ocr({ status: "error", text: null, confidence: null, error: { code: "OCR_TIMEOUT", message: "Время ожидания истекло" } });
  data.fields.push({ ...data.fields[0], name: "Face", label: "Фото", ocr: ocr({ status: "skipped", text: null, confidence: null, duration_seconds: 0 }) });
  data.meta.ocr = { model: "paddle", engine: "paddleocr", duration_seconds: 1.5, processed_fields: 2, failed_fields: 1 };
  const markup = renderToStaticMarkup(createElement(PassportFieldsView, { data }));
  assert.match(markup, /ПРИМЕР\nСТРОКА/);
  assert.match(markup, /Ошибка: Время ожидания истекло/);
  assert.match(markup, /OCR_TIMEOUT/);
  assert.match(markup, /Поле без печатного текста/);
  assert.match(markup, /Обработано полей: 2 · Ошибок: 1/);
  assert.equal((markup.match(/class="passport-field"/g) ?? []).length, 3);
});

test("explicit null OCR remains compatible and malformed OCR never renders as a successful result", () => {
  const data = sample();
  data.meta.ocr = null;
  data.fields[0].ocr = null;
  assert.notEqual(parsePassportFieldCrops(data), null);
  for (const invalid of [ocr({ status: "pending" }), ocr({ status: ["success"] }), ocr({ confidence: 3 }), ocr({ text: {} }), ocr({ duration_seconds: -1 })]) {
    data.fields[0].ocr = invalid;
    assert.equal(parsePassportFieldCrops(data), null);
  }
  data.fields[0].ocr = null;
  data.meta.ocr = { model: "paddle", engine: "paddleocr", duration_seconds: 1, processed_fields: 2, failed_fields: -1 };
  assert.equal(parsePassportFieldCrops(data), null);
});

test("a successful empty detection keeps the document and explicitly reports no fields", () => {
  const data = sample();
  data.fields = [];
  data.document.detected = false;
  data.document.orientation_confidence = null;
  const markup = renderToStaticMarkup(createElement(PassportFieldsView, { data }));
  assert.match(markup, /Поля не найдены/);
  assert.match(markup, /Без обрезки по границам документа/);
  assert.match(markup, /<image /);
  assert.doesNotMatch(markup, /class="passport-field"/);
});

test("unexpected, unsafe, or out-of-bounds results never become rendered image URLs or boxes", () => {
  for (const mutate of [
    (data) => { data.document.image = "https://other.example/document.jpg"; },
    (data) => { data.fields[0].image = "https://other.example/field.png"; },
    (data) => { data.fields[0].image = "data:image/svg+xml;base64,PHN2Zz4="; },
    (data) => { data.fields[0].image = "data:image/png;base64,"; },
    (data) => { data.fields[0].image = "data:image/png;base64,!!!!"; },
    (data) => { data.fields[0].bbox = [-1, 0, 200, 20]; },
    (data) => { data.fields[0].detection_bbox = [0, 0, 601, 20]; },
    (data) => { data.fields[0].confidence = Number.NaN; },
    (data) => { data.fields[0].confidence = 2; },
    (data) => { data.document.width = 0; },
    (data) => { data.fields = {}; },
  ]) {
    const data = sample();
    mutate(data);
    assert.equal(parsePassportFieldCrops(data), null);
    const markup = renderToStaticMarkup(createElement(PassportFieldsView, { data }));
    assert.match(markup, /неожиданный формат/);
    assert.doesNotMatch(markup, /<(?:img|image|svg)\b/);
  }
});

test("server labels are escaped as text", () => {
  const data = sample();
  data.fields[0].label = '<script>alert("x")</script>';
  const markup = renderToStaticMarkup(createElement(PassportFieldsView, { data }));
  assert.doesNotMatch(markup, /<script>/);
  assert.match(markup, /&lt;script&gt;/);
});

test("field recognition defaults to CPU Paddle and sends each selected OCR model as multipart", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  const tool = toolById["crop-passport-fields"];
  assert.deepEqual(defaultParams(tool), { ocr_model: "paddle", crop_document: true, padding: "4" });
  const selector = tool.params.find((param) => param.key === "ocr_model");
  assert.equal(selector.label, "Распознавание полей");
  assert.equal(selector.options.find((option) => option.value === "paddle").label, "Paddle eslav v5 · CPU");
  assert.equal(selector.options.find((option) => option.value === "paddle-cyrillic").label, "Paddle Cyrillic v5 · CPU");
  assert.equal(selector.options.find((option) => option.value === "russian-ocr-accurate").label, "Russian OCR accurate · CPU");
  assert.deepEqual(selector.options.map((option) => option.value), ["none", "paddle", "paddle-cyrillic", "russian-ocr-accurate", "qwen3-vl-4b", "qwen3-vl-8b"]);
  const file = new File(["synthetic"], "synthetic.jpg", { type: "image/jpeg" });
  const controller = new AbortController();
  for (const model of selector.options.map((option) => option.value)) {
    const params = { crop_document: false, padding: "16", ocr_model: model };
    globalThis.fetch = async (url, options) => {
      assert.equal(url, "/v1/ocr/crop/passport-fields");
      assert.equal(options.method, "POST");
      assert.equal(options.signal, controller.signal);
      assert.deepEqual([...options.body.keys()], ["image", "crop_document", "padding", "ocr_model"]);
      assert.equal(options.body.get("image").name, "synthetic.jpg");
      assert.equal(await options.body.get("image").text(), "synthetic");
      assert.equal(options.body.get("crop_document"), "false");
      assert.equal(options.body.get("padding"), "16");
      assert.equal(options.body.get("ocr_model"), model);
      return new Response(JSON.stringify({ success: true, data: sample(), error: null }), { headers: { "content-type": "application/json" } });
    };
    const response = await runTool("", tool, [file], params, controller.signal);
    assert.equal(response.ok, true);
    assert.equal(parsePassportFieldCrops(response.json.data).fields.length, 2);
    const curl = buildCurl("http://localhost:8000", tool, [file], params);
    assert.match(curl, /\/v1\/ocr\/crop\/passport-fields/);
    assert.match(curl, /-F 'crop_document=false'/);
    assert.match(curl, /-F 'padding=16'/);
    assert.ok(curl.includes(`-F 'ocr_model=${model}'`));
    assert.doesNotMatch(curl, /--output|normalize=/);
  }
});

test("new CPU alternatives are available only in the passport field crop experiment", () => {
  for (const tool of Object.values(toolById)) {
    if (tool.id === "crop-passport-fields") continue;
    const values = tool.params.flatMap((param) => param.kind === "select" ? param.options.map((option) => option.value) : []);
    assert.ok(!values.includes("paddle-cyrillic"), tool.id);
    assert.ok(!values.includes("russian-ocr-accurate"), tool.id);
  }
});
