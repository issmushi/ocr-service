export interface ReleaseNote {
  version: string;
  date: string;
  changes: string[];
}

export function parseChangelogResponse(payload: unknown): ReleaseNote[] {
  if (!payload || typeof payload !== "object" || !("releases" in payload)) {
    throw new Error("Сервер вернул историю изменений в неизвестном формате");
  }

  const releases = payload.releases;
  if (!Array.isArray(releases)) {
    throw new Error("Сервер вернул историю изменений в неизвестном формате");
  }

  return releases.map((release: unknown) => {
    if (!release || typeof release !== "object") {
      throw new Error("Сервер вернул историю изменений в неизвестном формате");
    }
    const entry = release as Record<string, unknown>;
    if (
      typeof entry.version !== "string" || !entry.version.trim()
      || typeof entry.date !== "string" || !entry.date.trim()
      || !Array.isArray(entry.changes)
      || !entry.changes.every((change: unknown) => typeof change === "string" && change.trim())
    ) {
      throw new Error("Сервер вернул историю изменений в неизвестном формате");
    }
    return {
      version: entry.version.trim(),
      date: entry.date.trim(),
      changes: entry.changes.map((change: string) => change.trim()),
    };
  });
}
