"""Serve short, manually maintained release notes to the internal UI."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter
from typing_extensions import TypedDict

CHANGELOG_PATH = Path(__file__).resolve().parents[2] / "CHANGELOG.md"
RELEASE_HEADING = re.compile(r"## (\d+\.\d+\.\d+) — (\d{4}-\d{2}-\d{2})")

router = APIRouter()


class ReleaseNote(TypedDict):
    version: str
    date: str
    changes: list[str]


def read_releases() -> list[ReleaseNote]:
    releases: list[ReleaseNote] = []
    current: ReleaseNote | None = None

    for line in CHANGELOG_PATH.read_text(encoding="utf-8").splitlines():
        heading = RELEASE_HEADING.fullmatch(line)
        if heading:
            current = {
                "version": heading.group(1),
                "date": heading.group(2),
                "changes": [],
            }
            releases.append(current)
        elif current is not None and line.startswith("- "):
            current["changes"].append(line[2:].strip())

    return releases


@router.get("/v1/changelog", include_in_schema=False)
async def changelog() -> dict[str, list[ReleaseNote]]:
    return {"releases": read_releases()}
