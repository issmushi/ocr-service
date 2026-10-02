import { useEffect, useState } from "react";
import { Icon } from "../components/Icon";
import { parseChangelogResponse } from "../changelog";
import type { ReleaseNote } from "../changelog";

type LoadState =
  | { status: "loading" }
  | { status: "ready"; releases: ReleaseNote[] }
  | { status: "error"; message: string };

function formatDate(value: string): string {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return value;
  const date = new Date(`${value}T12:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("ru-RU", {
    day: "numeric", month: "long", year: "numeric",
  }).format(date);
}

export function ReleaseList({ releases }: { releases: ReleaseNote[] }) {
  if (!releases.length) {
    return <p className="changelog-empty">Записей об обновлениях пока нет.</p>;
  }

  return (
    <div className="changelog-list">
      {releases.map((release, index) => (
        <article className={`release ${index === 0 ? "release--latest" : ""}`} key={`${release.version}-${release.date}`}>
          <div className="release__marker" aria-hidden="true" />
          <div className="release__card">
            <header className="release__head">
              <div>
                {index === 0 && <span className="release__latest">Последнее обновление</span>}
                <h2>Версия {release.version}</h2>
              </div>
              <time dateTime={release.date}>{formatDate(release.date)}</time>
            </header>
            <ul>
              {release.changes.map((change) => <li key={change}>{change}</li>)}
            </ul>
          </div>
        </article>
      ))}
    </div>
  );
}

export function ChangelogPage() {
  const [reloadKey, setReloadKey] = useState(0);
  const [state, setState] = useState<LoadState>({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    setState({ status: "loading" });

    async function load() {
      try {
        const response = await fetch("/v1/changelog", {
          headers: { Accept: "application/json" },
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`Историю изменений не удалось загрузить: HTTP ${response.status}`);
        const releases = parseChangelogResponse(await response.json());
        if (!controller.signal.aborted) setState({ status: "ready", releases });
      } catch (error) {
        if (controller.signal.aborted) return;
        setState({
          status: "error",
          message: error instanceof Error ? error.message : "Не удалось загрузить историю изменений",
        });
      }
    }

    void load();
    return () => controller.abort();
  }, [reloadKey]);

  return (
    <div className="page page--changelog">
      <header className="changelog-heading">
        <span className="eyebrow"><span className="eyebrow__dot" />Обновления сервиса</span>
        <h1>История изменений</h1>
        <p>Коротко о новых возможностях и изменениях в работе OCR.</p>
      </header>

      {state.status === "loading" && (
        <div aria-live="polite" className="changelog-state"><Icon name="refresh" size={20} className="spin" /><span>Загружаем обновления…</span></div>
      )}
      {state.status === "error" && (
        <div role="alert" className="changelog-state changelog-state--error">
          <Icon name="info" size={22} />
          <span>{state.message}</span>
          <button className="button button--secondary" onClick={() => setReloadKey((value) => value + 1)} type="button">Повторить</button>
        </div>
      )}
      {state.status === "ready" && <ReleaseList releases={state.releases} />}
    </div>
  );
}
