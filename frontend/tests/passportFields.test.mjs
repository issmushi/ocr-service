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
const { PassportFieldsView } = await import(await sourceUrl("../src/components/PassportFieldsView.tsx", {
  "./passportFields": parserUrl,
  react: pathToFileURL(require.resolve("react")).href,
  "react/jsx-runtime": pathToFileURL(require.resolve("react/jsx-runtime")).href,
}));
const { toolById, defaultParams } = await import(await sourceUrl("../src/config.ts"));
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
    (data) => { data.fields[0].image = "data:image/svg+xml;base64,PHN2Zz4="; },
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

test("the field detector sends exactly the selected image, crop flag, and padding as multipart", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  const tool = toolById["crop-passport-fields"];
  assert.deepEqual(defaultParams(tool), { crop_document: true, padding: "4" });
  const params = { crop_document: false, padding: "16" };
  const file = new File(["synthetic"], "synthetic.jpg", { type: "image/jpeg" });
  const controller = new AbortController();
  globalThis.fetch = async (url, options) => {
    assert.equal(url, "/v1/ocr/crop/passport-fields");
    assert.equal(options.method, "POST");
    assert.equal(options.signal, controller.signal);
    assert.deepEqual([...options.body.keys()], ["image", "crop_document", "padding"]);
    assert.equal(options.body.get("image").name, "synthetic.jpg");
    assert.equal(await options.body.get("image").text(), "synthetic");
    assert.equal(options.body.get("crop_document"), "false");
    assert.equal(options.body.get("padding"), "16");
    return new Response(JSON.stringify({ success: true, data: sample(), error: null }), { headers: { "content-type": "application/json" } });
  };
  const response = await runTool("", tool, [file], params, controller.signal);
  assert.equal(response.ok, true);
  assert.equal(parsePassportFieldCrops(response.json.data).fields.length, 2);
  const curl = buildCurl("http://localhost:8000", tool, [file], params);
  assert.match(curl, /\/v1\/ocr\/crop\/passport-fields/);
  assert.match(curl, /-F 'crop_document=false'/);
  assert.match(curl, /-F 'padding=16'/);
  assert.doesNotMatch(curl, /--output|model=|normalize=/);
});
