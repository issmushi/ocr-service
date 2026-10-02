import { useCallback, useEffect, useState } from "react";
import { resolveApiUrl } from "./api";
import { tools } from "./config";
import { DocsPage } from "./pages/DocsPage";
import { ChangelogPage } from "./pages/ChangelogPage";
import { OverviewPage } from "./pages/OverviewPage";
import { PlaygroundPage } from "./pages/PlaygroundPage";
import type { HealthState, Section, ToolId } from "./types";
import { Icon } from "./components/Icon";

const sections: { id: Section; label: string; description: string }[] = [
  { id: "overview", label: "Обзор", description: "Эндпоинты и обработка" },
  { id: "docs", label: "Документация", description: "Параметры и примеры" },
  { id: "playground", label: "Тестирование", description: "Запросы к API" },
  { id: "changelog", label: "История изменений", description: "Обновления по версиям" },
];

const sectionIcons = {
  overview: "layers",
  docs: "book",
  playground: "play",
  changelog: "clock",
} as const;

const uiBasePath = import.meta.env.BASE_URL.replace(/\/+$/, "");

function getRouteParts(): string[] {
  const hash = window.location.hash.startsWith("#/")
    ? window.location.hash.slice(2)
    : "";
  if (hash) return hash.split("/").filter(Boolean);

  const pathname = window.location.pathname;
  const relative = pathname === uiBasePath
    ? ""
    : pathname.startsWith(`${uiBasePath}/`)
      ? pathname.slice(uiBasePath.length + 1)
      : "";
  return relative.split("/").filter(Boolean);
}

function getSection(): Section {
  const section = getRouteParts()[0];
  return sections.some((item) => item.id === section) ? (section as Section) : "overview";
}

function getInitialTool(): ToolId {
  const value = getRouteParts()[1];
  return tools.some((tool) => tool.id === value) ? (value as ToolId) : "mrz";
}

export function App() {
  const [section, setSection] = useState<Section>(getSection);
  const [initialTool, setInitialTool] = useState<ToolId>(getInitialTool);
  const [mobileNav, setMobileNav] = useState(false);
  const [isMobile, setIsMobile] = useState(() => window.matchMedia("(max-width: 820px)").matches);
  const apiBaseUrl = "";
  const [health, setHealth] = useState<HealthState>({ status: "idle" });

  const navigate = useCallback((next: Section, tool?: ToolId) => {
    const route = tool ? `${next}/${tool}` : next;
    window.history.pushState(null, "", `${uiBasePath}/${route}`);
    setSection(next);
    if (tool) setInitialTool(tool);
    setMobileNav(false);
    window.scrollTo({ top: 0, left: 0, behavior: "smooth" });
  }, []);

  useEffect(() => {
    const onHash = () => {
      setSection(getSection());
      setInitialTool(getInitialTool());
    };
    window.addEventListener("hashchange", onHash);
    window.addEventListener("popstate", onHash);
    return () => {
      window.removeEventListener("hashchange", onHash);
      window.removeEventListener("popstate", onHash);
    };
  }, []);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 820px)");
    const onChange = (event: MediaQueryListEvent) => {
      setIsMobile(event.matches);
      if (!event.matches) setMobileNav(false);
    };
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    if (!mobileNav) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileNav(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [mobileNav]);

  const checkHealth = useCallback(async () => {
    setHealth({ status: "checking" });
    const started = performance.now();
    try {
      const response = await fetch(resolveApiUrl(apiBaseUrl, "/health"), {
        headers: { Accept: "application/json" },
      });
      const latencyMs = Math.round(performance.now() - started);
      const payload = (await response.json().catch(() => null)) as unknown;
      const object = payload && typeof payload === "object" ? (payload as Record<string, unknown>) : null;
      const data = object?.data && typeof object.data === "object" ? object.data as Record<string, unknown> : object;
      const status = data?.status;
      const version = typeof data?.version === "string" && data.version.trim() && data.version !== "unknown"
        ? data.version.trim()
        : undefined;
      if (status === "degraded") {
        setHealth({ status: "degraded", version, latencyMs, message: "Часть компонентов недоступна" });
      } else if (!response.ok) {
        setHealth({ status: "offline", version, latencyMs, message: `HTTP ${response.status}` });
      } else if (status && status !== "ok") {
        setHealth({ status: "offline", version, latencyMs, message: String(status) });
      } else {
        setHealth({ status: "online", version, latencyMs });
      }
    } catch (error) {
      setHealth({
        status: "offline",
        message: error instanceof Error ? error.message : "Нет соединения",
      });
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    void checkHealth();
  }, [checkHealth]);

  const baseLabel = apiBaseUrl || "этот сервер";

  return (
    <div className="app-shell">
      <aside
        aria-hidden={isMobile && !mobileNav}
        className={`sidebar ${mobileNav ? "sidebar--open" : ""}`}
        id="primary-sidebar"
        inert={isMobile && !mobileNav}
      >
        <div className="brand">
          <div className="brand__mark"><Icon name="scan" size={22} /></div>
          <div>
            <strong>OCR Service</strong>
            <span>Документация и тесты</span>
          </div>
        </div>

        <nav className="primary-nav" aria-label="Основная навигация">
          <span className="nav-kicker">Разделы</span>
          {sections.map((item) => (
            <button
              className={`nav-item ${section === item.id ? "nav-item--active" : ""}`}
              key={item.id}
              onClick={() => navigate(item.id)}
              type="button"
            >
              <Icon name={sectionIcons[item.id]} />
              <span><strong>{item.label}</strong><small>{item.description}</small></span>
            </button>
          ))}
        </nav>

        <div className="sidebar__footer">
          <div className="api-mini-status">
            <span className={`status-dot status-dot--${health.status}`} />
            <div>
              <strong>{health.status === "online" ? "API доступен" : health.status === "degraded" ? "API работает частично" : health.status === "checking" ? "Проверка API" : health.status === "offline" ? "API недоступен" : "API"}</strong>
              <span>{health.latencyMs ? `${health.latencyMs} мс` : baseLabel}</span>
            </div>
          </div>
          <span className="build-label">Версия {health.version ?? "—"}</span>
        </div>
      </aside>
      {mobileNav && <button className="sidebar-backdrop" aria-label="Закрыть меню" onClick={() => setMobileNav(false)} />}

      <div className="workspace">
        <header className="topbar">
          <button
            aria-controls="primary-sidebar"
            aria-expanded={mobileNav}
            aria-label="Открыть меню"
            className="mobile-menu"
            onClick={() => setMobileNav(true)}
          >
            <Icon name="menu" />
          </button>
          <div className="topbar__path">
            <span>OCR</span><Icon name="chevron" size={15} /><strong>{sections.find((item) => item.id === section)?.label}</strong>
          </div>
        </header>

        <main>
          {section === "overview" && <OverviewPage health={health} navigate={navigate} />}
          {section === "docs" && <DocsPage navigate={navigate} />}
          {section === "changelog" && <ChangelogPage />}
          {section === "playground" && (
            <PlaygroundPage
              apiBaseUrl={apiBaseUrl}
              initialTool={initialTool}
              onToolChange={(tool) => {
                setInitialTool(tool);
                window.history.replaceState(null, "", `${uiBasePath}/playground/${tool}`);
              }}
            />
          )}
        </main>
      </div>
    </div>
  );
}
