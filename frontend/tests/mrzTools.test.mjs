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

const configUrl = await sourceUrl("../src/config.ts");
const { tools, toolById, defaultParams } = await import(configUrl);
const { runTool, buildCurl } = await import(await sourceUrl("../src/api.ts"));
const reactImports = {
  react: pathToFileURL(require.resolve("react")).href,
  "react/jsx-runtime": pathToFileURL(require.resolve("react/jsx-runtime")).href,
};
const pageImports = {
  ...reactImports,
  "../config": configUrl,
  "../components/Icon": await sourceUrl("../src/components/Icon.tsx", reactImports),
};
const { FieldsView } = await import(await sourceUrl("../src/components/FieldsView.tsx", reactImports));
const { DocsPage } = await import(await sourceUrl("../src/pages/DocsPage.tsx", pageImports));
const { OverviewPage } = await import(await sourceUrl("../src/pages/OverviewPage.tsx", pageImports));

test("unified MRZ tool sends only image and processing flags, with no model selector or cURL parameter", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  const file = new File(["synthetic"], "document.jpg", { type: "image/jpeg" });
  const controller = new AbortController();

  const tool = toolById.mrz;
  assert.equal(tool.endpoint, "/v1/ocr/mrz");
  assert.equal(tool.group, "MRZ");
  assert.deepEqual(tools.filter((item) => item.group === "MRZ").map((item) => item.id), ["mrz"]);
  assert.equal(tools.some((item) => item.id === "passport-mrz" || item.id === "idcard-mrz"), false);
  assert.deepEqual(defaultParams(tool), { crop: true, normalize: true, normalization: true });
  assert.deepEqual(tool.params.map((param) => param.key), ["crop", "normalize", "normalization"]);
  assert.ok(tool.params.every((param) => param.kind === "toggle"));
  assert.match(tool.description, /MRZ Scanner/);

  for (const params of [defaultParams(tool), { crop: false, normalize: false, normalization: false }]) {
    globalThis.fetch = async (url, options) => {
      assert.equal(url, "/v1/ocr/mrz");
      assert.equal(options.method, "POST");
      assert.equal(options.signal, controller.signal);
      assert.deepEqual([...options.body.keys()], ["image", "crop", "normalize", "normalization"]);
      assert.equal(await options.body.get("image").text(), "synthetic");
      for (const [key, value] of Object.entries(params)) assert.equal(options.body.get(key), String(value));
      assert.equal(options.body.has("model"), false);
      return new Response(JSON.stringify({ success: true, data: { mrz: [] }, error: null }), { headers: { "content-type": "application/json" } });
    };
    assert.equal((await runTool("", tool, [file], params, controller.signal)).ok, true);
    const curl = buildCurl("http://localhost:8000", tool, [file], params);
    assert.ok(curl.includes(tool.endpoint));
    assert.doesNotMatch(curl, /model=/);
    for (const [key, value] of Object.entries(params)) assert.ok(curl.includes(`-F '${key}=${value}'`));
  }
});

test("internal passport sends only image and uses Russian OCR", async (t) => {
  const tool = toolById["passport-internal"];
  assert.equal(tool.endpoint, "/v1/ocr/passport/internal");
  assert.match(tool.description, /Russian OCR/);
  assert.deepEqual(tool.params, []);
  assert.deepEqual(defaultParams(tool), {});

  const file = new File(["synthetic"], "passport.jpg", { type: "image/jpeg" });
  const originalFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  globalThis.fetch = async (url, options) => {
    assert.equal(url, "/v1/ocr/passport/internal");
    assert.equal(options.method, "POST");
    assert.deepEqual([...options.body.keys()], ["image"]);
    assert.equal(await options.body.get("image").text(), "synthetic");
    return new Response(JSON.stringify({ success: true, data: { Issue_organization_ru: "МВД России" }, error: null }), { headers: { "content-type": "application/json" } });
  };
  const result = await runTool("", tool, [file], defaultParams(tool), new AbortController().signal);
  assert.equal(result.ok, true);
  assert.equal(result.json.data.Issue_organization_ru, "МВД России");

  const curl = buildCurl("http://localhost:8000", tool, [file], defaultParams(tool));
  assert.match(curl, /\/v1\/ocr\/passport\/internal/);
  assert.match(curl, /image=@/);
  assert.doesNotMatch(curl, /-F '(?:page|model|crop)=/);

  const docs = renderToStaticMarkup(createElement(DocsPage, { navigate() {} }));
  assert.match(docs, /\/v1\/tasks\/passport\/internal/);
  assert.match(docs, /Issue_organization_ru/);
  assert.match(docs, /Кем выдан/);
  assert.match(docs, /Поле.*mrz.*содержит две строки.*иначе.*null/);
});

test("Russian international passport sends only image and exposes its own task route", async (t) => {
  const tool = toolById["passport-international"];
  assert.equal(tool.endpoint, "/v1/ocr/passport/international");
  assert.equal(tool.group, "Россия");
  assert.match(tool.description, /Russian OCR/);
  assert.deepEqual(tool.params, []);
  assert.deepEqual(defaultParams(tool), {});

  const file = new File(["synthetic"], "passport.jpg", { type: "image/jpeg" });
  const originalFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  globalThis.fetch = async (url, options) => {
    assert.equal(url, "/v1/ocr/passport/international");
    assert.equal(options.method, "POST");
    assert.deepEqual([...options.body.keys()], ["image"]);
    return new Response(JSON.stringify({ success: true, data: { passport_type: "international", mrz: null }, error: null }), { headers: { "content-type": "application/json" } });
  };
  assert.equal((await runTool("", tool, [file], {}, new AbortController().signal)).ok, true);

  const curl = buildCurl("http://localhost:8000", tool, [file], {});
  assert.match(curl, /\/v1\/ocr\/passport\/international/);
  assert.doesNotMatch(curl, /-F '(?:page|model|crop)=/);

  const docs = renderToStaticMarkup(createElement(DocsPage, { navigate() {} }));
  assert.match(docs, /\/v1\/tasks\/passport\/international/);
  assert.match(docs, /issue_organization_code/);
  assert.match(docs, /surname_latin/);
  assert.match(docs, /девятизначный/);
  assert.match(docs, /Отдельной серии паспорта здесь нет/);
  assert.match(docs, /Загранпаспорт РФ/);

  const fields = renderToStaticMarkup(createElement(FieldsView, {
    entries: [["surname_latin", "IVANOV"], ["birth_place_ru", "МОСКВА"], ["issue_organization_code", "77001"]],
  }));
  assert.match(fields, /Фамилия \(латиницей\)/);
  assert.match(fields, /Место рождения \(RU\)/);
  assert.match(fields, /Код органа выдачи/);
});

test("SNILS sends one image, shows its fields, and documents the task route", async (t) => {
  const tool = toolById.snils;
  assert.equal(tool.endpoint, "/v1/ocr/snils");
  assert.equal(tool.group, "Россия");
  assert.match(tool.description, /Paddle Cyrillic/);
  assert.deepEqual(tool.params, []);
  assert.deepEqual(defaultParams(tool), {});

  const file = new File(["synthetic"], "snils.jpg", { type: "image/jpeg" });
  const originalFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  globalThis.fetch = async (url, options) => {
    assert.equal(url, "/v1/ocr/snils");
    assert.equal(options.method, "POST");
    assert.deepEqual([...options.body.keys()], ["image"]);
    return new Response(JSON.stringify({
      success: true,
      data: { snils_number: "000-000-000 00", date_of_birth: "01.01.2000", meta: { engine: "russian-ocr" } },
      error: null,
    }), { headers: { "content-type": "application/json" } });
  };
  const result = await runTool("", tool, [file], defaultParams(tool), new AbortController().signal);
  assert.equal(result.ok, true);
  assert.equal(result.json.data.date_of_birth, "01.01.2000");

  const curl = buildCurl("http://localhost:8000", tool, [file], {});
  assert.match(curl, /\/v1\/ocr\/snils/);
  assert.match(curl, /image=@/);
  assert.doesNotMatch(curl, /-F '(?:model|crop)=/);

  const docs = renderToStaticMarkup(createElement(DocsPage, { navigate() {} }));
  assert.match(docs, /\/v1\/tasks\/snils/);
  assert.match(docs, /Paddle Cyrillic/);
  assert.match(docs, /date_of_birth/);

  const fields = renderToStaticMarkup(createElement(FieldsView, {
    entries: [["snils_number", "000-000-000 00"], ["birth_place", "ГОРОД"], ["registration_date", "02.01.2000"]],
  }));
  assert.match(fields, /Номер СНИЛС/);
  assert.match(fields, /Место рождения/);
  assert.match(fields, /Дата регистрации/);
});

test("MRZ fields show the shared document names and all document types", () => {
  for (const [type, label] of [
    ["internal_passport", "Внутренний паспорт"],
    ["international_passport", "Заграничный паспорт"],
    ["identity_card", "ID-карта"],
  ]) {
    const markup = renderToStaticMarkup(createElement(FieldsView, {
      entries: [["document_type", type], ["document_series", null], ["document_number", "123456789"]],
    }));
    assert.match(markup, /Тип документа/);
    assert.ok(markup.includes(label));
    assert.match(markup, /Серия документа/);
    assert.match(markup, /Не найдено/);
    assert.match(markup, /Номер документа/);
    assert.match(markup, /123456789/);
    assert.doesNotMatch(markup, /Тип паспорта|Серия паспорта|Номер паспорта/);
  }
});

test("full text and field crop tools retain model selection", () => {
  for (const [id, key, expectedDefault] of [
    ["full-ocr", "model", "paddle"],
    ["crop-passport-fields", "ocr_model", "paddle"],
  ]) {
    const tool = toolById[id];
    const selector = tool.params.find((param) => param.key === key);
    assert.equal(selector.kind, "select");
    assert.ok(selector.options.some((option) => option.value === "qwen3-vl-4b"));
    assert.equal(defaultParams(tool)[key], expectedDefault);
    const curl = buildCurl("http://localhost:8000", tool, [], defaultParams(tool));
    assert.ok(curl.includes(`-F '${key}=${expectedDefault}'`));
  }
});

test("documentation and overview show MRZ Scanner and examples without model", () => {
  const docs = renderToStaticMarkup(createElement(DocsPage, { navigate() {} }));
  const example = docs.match(/<pre><code>([\s\S]*?)<\/code><\/pre>/)?.[1];
  assert.ok(example);
  assert.match(example, /\/v1\/ocr\/mrz/);
  assert.doesNotMatch(example, /model=/);
  assert.match(docs, /\/v1\/tasks\/mrz/);
  assert.doesNotMatch(docs, /\/v1\/(?:ocr|tasks)\/(?:passport|idcard)\/mrz/);
  for (const field of ["document_type", "document_series", "document_number"]) assert.match(docs, new RegExp(field));
  for (const documentType of ["internal_passport", "international_passport", "identity_card"]) assert.match(docs, new RegExp(documentType));
  assert.match(docs, /Единый MRZ-эндпоинт использует MRZ Scanner/);
  assert.match(docs, /MRZ Scanner распознаёт строки паспорта или ID-карты/);

  const overview = renderToStaticMarkup(createElement(OverviewPage, { health: { status: "idle" }, navigate() {} }));
  const request = overview.match(/class="hero-console__body">([\s\S]*?)<div class="console-divider"/)?.[1];
  assert.ok(request);
  assert.doesNotMatch(request, />model</);
  assert.match(request, />crop</);
  assert.match(overview, /POST \/v1\/ocr\/mrz/);
  assert.doesNotMatch(overview, /\/v1\/ocr\/(?:passport|idcard)\/mrz/);
  assert.match(overview, /MRZ Scanner читает строки MRZ паспорта или ID-карты/);
  assert.doesNotMatch(overview, /Чтение строк через PaddleOCR|MRZ Scanner или Qwen3-VL/);
});
