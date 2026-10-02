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
  for (const [specifier, url] of Object.entries(imports)) {
    outputText = outputText.replaceAll(`from "${specifier}"`, `from "${url}"`);
  }
  return `data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`;
}

const changelogUrl = await sourceUrl("../src/changelog.ts");
const { parseChangelogResponse } = await import(changelogUrl);
const reactImports = {
  react: pathToFileURL(require.resolve("react")).href,
  "react/jsx-runtime": pathToFileURL(require.resolve("react/jsx-runtime")).href,
};
const { ReleaseList } = await import(await sourceUrl("../src/pages/ChangelogPage.tsx", {
  ...reactImports,
  "../changelog": changelogUrl,
  "../components/Icon": await sourceUrl("../src/components/Icon.tsx", reactImports),
}));

test("history renders concise release notes from the API contract", () => {
  const releases = parseChangelogResponse({ releases: [
    { version: "1.1.0", date: "2026-10-02", changes: ["Добавлено распознавание документа."] },
    { version: "1.0.0", date: "2026-09-20", changes: ["Первый выпуск API."] },
  ] });
  const markup = renderToStaticMarkup(createElement(ReleaseList, { releases }));
  assert.match(markup, /Последнее обновление/);
  assert.match(markup, /Версия 1\.1\.0/);
  assert.match(markup, /Версия 1\.0\.0/);
  assert.match(markup, /Добавлено распознавание документа/);
  assert.match(markup, /Первый выпуск API/);
  assert.match(markup, /2026-10-02/);
});

test("history rejects malformed releases instead of silently showing an empty list", () => {
  assert.throws(() => parseChangelogResponse({}), /неизвестном формате/);
  assert.throws(() => parseChangelogResponse({ releases: [
    { version: "1.1.0", date: "2026-10-02", changes: "текст" },
  ] }), /неизвестном формате/);
});
