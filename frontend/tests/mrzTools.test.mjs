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
const { toolById, defaultParams } = await import(configUrl);
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
const { DocsPage } = await import(await sourceUrl("../src/pages/DocsPage.tsx", pageImports));
const { OverviewPage } = await import(await sourceUrl("../src/pages/OverviewPage.tsx", pageImports));

test("both MRZ tools send only image and processing flags, with no model selector or cURL parameter", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  const file = new File(["synthetic"], "document.jpg", { type: "image/jpeg" });
  const controller = new AbortController();

  for (const id of ["passport-mrz", "idcard-mrz"]) {
    const tool = toolById[id];
    assert.deepEqual(defaultParams(tool), { crop: true, normalize: true, normalization: true });
    assert.deepEqual(tool.params.map((param) => param.key), ["crop", "normalize", "normalization"]);
    assert.ok(tool.params.every((param) => param.kind === "toggle"));
    assert.match(tool.description, /MRZ Scanner/);

    for (const params of [defaultParams(tool), { crop: false, normalize: false, normalization: false }]) {
      globalThis.fetch = async (url, options) => {
        assert.equal(url, tool.endpoint);
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
  assert.match(example, /\/v1\/ocr\/passport\/mrz/);
  assert.doesNotMatch(example, /model=/);
  assert.match(docs, /Оба MRZ-эндпоинта используют MRZ Scanner/);
  assert.match(docs, /MRZ Scanner распознаёт строки паспорта или ID-карты/);

  const overview = renderToStaticMarkup(createElement(OverviewPage, { health: { status: "idle" }, navigate() {} }));
  const request = overview.match(/class="hero-console__body">([\s\S]*?)<div class="console-divider"/)?.[1];
  assert.ok(request);
  assert.doesNotMatch(request, />model</);
  assert.match(request, />crop</);
  assert.match(overview, /MRZ Scanner читает строки MRZ паспорта или ID-карты/);
  assert.doesNotMatch(overview, /Чтение строк через PaddleOCR|MRZ Scanner или Qwen3-VL/);
});
