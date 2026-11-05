import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

const require = createRequire(import.meta.url);
async function sourceUrl(path, imports = {}) {
  const source = await readFile(new URL(path, import.meta.url), "utf8");
  let { outputText } = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022, jsx: ts.JsxEmit.ReactJSX },
  });
  for (const [specifier, url] of Object.entries(imports)) outputText = outputText.replaceAll(`from "${specifier}"`, `from "${url}"`);
  return `data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`;
}
const fieldOcrViewUrl = await sourceUrl("../src/components/FieldOcrView.tsx", {
  "../config": await sourceUrl("../src/config.ts"),
  "react/jsx-runtime": pathToFileURL(require.resolve("react/jsx-runtime")).href,
});
const { EgrnFieldsView, parseEgrnFields } = await import(await sourceUrl("../src/components/EgrnFieldsView.tsx", {
  "./FieldOcrView": fieldOcrViewUrl,
  "./passportFields": await sourceUrl("../src/components/passportFields.ts"),
  "../config": await sourceUrl("../src/config.ts"),
  react: pathToFileURL(require.resolve("react")).href,
  "react/jsx-runtime": pathToFileURL(require.resolve("react/jsx-runtime")).href,
}));
const { toolById, defaultParams } = await import(await sourceUrl("../src/config.ts"));
const { runTool } = await import(await sourceUrl("../src/api.ts"));
const names = ["application_number", "cadastral_number", "address", "object_name", "actuality_date"];
const image = "data:image/png;base64,iVBORw0KGgo=";
function sample() {
  return {
    document: { image, width: 600, height: 500 },
    fields: names.map((name, index) => ({ name, label: `Область ${index + 1}`, image, width: 200, height: 30, bbox: [100, 50 + index * 70, 300, 80 + index * 70] })),
    missing_fields: [],
    meta: { engine: "opencv", model: "egrn-gosuslugi-first-page", duration_seconds: 0.42 },
  };
}
test("EGRN shows five exact overlays and independent PNG downloads", () => {
  const result = sample();
  const markup = renderToStaticMarkup(createElement(EgrnFieldsView, { data: result }));
  assert.match(markup, /Найдено участков: 5 из 5/);
  assert.match(markup, /viewBox="0 0 600 500"/);
  assert.match(markup, /height="30"[^>]*width="200" x="100" y="50"/);
  for (const name of names) assert.match(markup, new RegExp(`download="egrn-${name}\\.png"`));
  assert.doesNotMatch(markup, /Уверенность|100%/);
});
test("missing regions remain explicit instead of fabricated crops", () => {
  const result = sample();
  result.fields = result.fields.filter((field) => field.name !== "address");
  result.missing_fields = ["address"];
  assert.ok(parseEgrnFields(result));
  const markup = renderToStaticMarkup(createElement(EgrnFieldsView, { data: result }));
  assert.match(markup, /Найдено участков: 4 из 5/);
  assert.match(markup, /Не найдены:/);
  assert.match(markup, /Адрес \(местоположение\)/);
  assert.doesNotMatch(markup, /download="egrn-address/);
});
test("malformed or external image payloads are not rendered", () => {
  for (const alter of [
    (r) => { r.document.image = "https://external.invalid/scan.png"; },
    (r) => { r.fields[0].bbox[2] = 900; },
    (r) => { r.fields[0].width = 201; },
    (r) => { r.fields.push(r.fields[0]); },
    (r) => { r.fields[0].name = "unknown"; },
    (r) => { r.fields[0].image = "data:image/svg+xml;base64,PHN2Zz4="; },
    (r) => { r.meta.duration_seconds = NaN; },
  ]) {
    const result = sample(); alter(result);
    assert.equal(parseEgrnFields(result), null);
  }
});
test("EGRN fields tool sends the chosen OCR model and receives the JSON envelope", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  globalThis.fetch = async (url, options) => {
    assert.equal(url, "/v1/ocr/crop/egrn-fields");
    assert.deepEqual([...options.body.keys()], ["image", "ocr_model"]);
    assert.equal(options.body.get("ocr_model"), "paddle");
    return Response.json({ success: true, data: sample(), error: null });
  };
  const tool = toolById["crop-egrn-fields"];
  const result = await runTool("", tool, [new File(["synthetic"], "scan.png")], defaultParams(tool), new AbortController().signal);
  assert.equal(result.ok, true);
  assert.ok(parseEgrnFields(result.json.data));
});

test("EGRN displays raw multiline OCR, failures and the same model labels as document fields", () => {
  const result = sample();
  const ocr = { model: "paddle-cyrillic", engine: "paddleocr", device: "cpu", status: "success", text: "ТЕСТОВЫЙ АДРЕС\nСТРОКА 2", confidence: 0.93, duration_seconds: 0.2, error: null };
  result.fields[2].ocr = ocr;
  result.fields[3].ocr = { ...ocr, status: "error", text: null, confidence: null, error: { code: "OCR_FAILED", message: "Ошибка поля" } };
  result.meta.ocr = { model: "paddle-cyrillic", engine: "paddleocr", duration_seconds: 1, processed_fields: 5, failed_fields: 1 };
  assert.ok(parseEgrnFields(result));
  const markup = renderToStaticMarkup(createElement(EgrnFieldsView, { data: result }));
  assert.match(markup, /ТЕСТОВЫЙ АДРЕС\nСТРОКА 2/);
  assert.match(markup, /Paddle Cyrillic v5/);
  assert.match(markup, /Ошибок: 1/);
  assert.match(markup, /Ошибка поля/);
  assert.match(markup, /93.0%/);
  result.fields[0].ocr = { ...ocr, confidence: 2 };
  assert.equal(parseEgrnFields(result), null);
});
