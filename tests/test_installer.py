from __future__ import annotations

import os
import pty
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "install.sh"


def run_installer(
    *arguments: str,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    return subprocess.run(
        ["bash", str(INSTALLER), *arguments],
        cwd=ROOT,
        env=command_env,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def run_interactive_installer(
    *arguments: str,
    input_text: str,
) -> subprocess.CompletedProcess[str]:
    master_fd, slave_fd = pty.openpty()
    try:
        os.write(master_fd, input_text.encode())
        return subprocess.run(
            ["bash", str(INSTALLER), *arguments],
            cwd=ROOT,
            env=os.environ.copy(),
            stdin=slave_fd,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_standard_cpu_dry_run_is_read_only() -> None:
    env_path = ROOT / ".env"
    before = env_path.read_bytes() if env_path.exists() else None

    result = run_installer(
        "--edition",
        "standard",
        "--device",
        "cpu",
        "--yes",
        "--dry-run",
    )

    after = env_path.read_bytes() if env_path.exists() else None
    assert result.returncode == 0, result.stderr
    assert "paddle,mrz,russian-ocr" in result.stdout
    assert "система не изменена" in result.stdout
    assert after == before


def test_dry_run_does_not_fetch_russian_ocr_assets(
    tmp_path: Path,
) -> None:
    target = tmp_path / "models"
    result = run_installer(
        "--edition",
        "standard",
        "--device",
        "cpu",
        "--yes",
        "--dry-run",
        env={
            "RUSSIAN_OCR_MODELS_DIR": str(target),
            "RUSSIAN_OCR_MODELS_SOURCE_DIR": str(tmp_path / "missing"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert not target.exists()


def test_manual_edition_reads_models_from_environment() -> None:
    result = run_installer(
        "--edition",
        "manual",
        "--device",
        "cpu",
        "--yes",
        "--dry-run",
        env={"OCR_MODELS": "paddle,mrz"},
    )

    assert result.returncode == 0, result.stderr
    assert "Комплектация:  manual" in result.stdout
    assert "Модели:        paddle,mrz" in result.stdout


@pytest.mark.parametrize("edition", ["standard", "extended"])
def test_interactive_device_menu_hides_unsupported_npu(edition: str) -> None:
    result = run_interactive_installer(
        "--edition",
        edition,
        "--yes",
        "--dry-run",
        input_text="1\n",
    )

    assert result.returncode == 0, result.stderr
    assert "Выбери режим вычислений" in result.stdout
    assert "NPU" not in result.stdout
    assert "Режим:         CPU" in result.stdout


def test_help_marks_explicit_npu_as_diagnostic_only() -> None:
    result = run_installer("--help")

    assert result.returncode == 0
    assert "NPU — только явная диагностическая проверка" in result.stdout


def test_npu_selection_fails_before_installation() -> None:
    env_path = ROOT / ".env"
    before = env_path.read_bytes() if env_path.exists() else None

    result = run_installer(
        "--edition",
        "standard",
        "--device",
        "npu",
        "--yes",
    )

    after = env_path.read_bytes() if env_path.exists() else None
    assert result.returncode != 0
    assert "Установка не начата" in result.stderr
    assert after == before


def test_linux_arm_cpu_profile_is_rejected_before_installation(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uname = fake_bin / "uname"
    fake_uname.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  -s) echo Linux ;;\n"
        "  -m) echo aarch64 ;;\n"
        "  *) echo Linux ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_uname.chmod(0o755)

    result = run_installer(
        "--edition",
        "standard",
        "--device",
        "cpu",
        "--yes",
        "--dry-run",
        env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert result.returncode != 0
    assert "Linux/aarch64" in result.stderr
    assert "Установка не начата" in result.stderr


def test_installer_is_executable() -> None:
    assert INSTALLER.stat().st_mode & 0o111


def test_offline_russian_ocr_assets_are_fetched_then_checked(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    fake_bin = tmp_path / "bin"
    project.mkdir()
    fake_bin.mkdir()
    shutil.copy2(INSTALLER, project / "install.sh")
    (project / "profiles" / "cpu").mkdir(parents=True)
    (project / "frontend").mkdir()
    (project / "frontend" / "package.json").write_text(
        '{"scripts":{"build":"test-build"}}',
        encoding="utf-8",
    )
    (project / "frontend" / "package-lock.json").write_text(
        "{}", encoding="utf-8"
    )

    call_log = tmp_path / "python-calls.log"
    fake_python = fake_bin / "python3.11"
    fake_python.write_text(
        '#!/bin/sh\nif [ "${1:-}" = "-c" ]; then echo 3.11; fi\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    fake_node = fake_bin / "node"
    fake_node.write_text(
        '#!/bin/sh\nif [ "${1:-}" = "-p" ]; then echo 20; fi\n',
        encoding="utf-8",
    )
    fake_node.chmod(0o755)

    fake_npm = fake_bin / "npm"
    fake_npm.write_text(
        "#!/bin/sh\n"
        'if [ "${1:-}" = "run" ] && [ "${2:-}" = "build" ]; then\n'
        '  out_dir="$PWD/../app/static/ui"\n'
        '  while [ "$#" -gt 0 ]; do\n'
        '    if [ "$1" = "--outDir" ]; then out_dir=$2; shift 2; '
        "else shift; fi\n"
        "  done\n"
        '  mkdir -p "$out_dir"\n'
        "  printf '<!doctype html>' > \"$out_dir/index.html\"\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_npm.chmod(0o755)

    fake_poetry = fake_bin / "poetry"
    fake_poetry.write_text(
        "#!/bin/sh\n"
        "profile=\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-C" ]; then\n'
        "    profile=$2\n"
        "    shift 2\n"
        "  else\n"
        "    shift\n"
        "  fi\n"
        "done\n"
        'mkdir -p "$profile/.venv/bin"\n'
        "cat > \"$profile/.venv/bin/python\" <<'PY'\n"
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "$INSTALL_TEST_LOG"\n'
        'if [ "${1:-}" = "-c" ]; then echo test-key; fi\n'
        "PY\n"
        'chmod +x "$profile/.venv/bin/python"\n',
        encoding="utf-8",
    )
    fake_poetry.chmod(0o755)

    source = tmp_path / "offline-models"
    target = tmp_path / "installed-models"
    source.mkdir()
    command_env = os.environ.copy()
    command_env.update(
        {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "INSTALL_TEST_LOG": str(call_log),
            "OCR_MODELS": "russian-ocr",
            "RUSSIAN_OCR_MODELS_DIR": str(target),
            "RUSSIAN_OCR_MODELS_SOURCE_DIR": str(source),
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(project / "install.sh"),
            "--edition",
            "manual",
            "--device",
            "cpu",
            "--yes",
            "--no-start",
        ],
        cwd=project,
        env=command_env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    fetch = (
        "-m app.ocr_engine.russian.assets fetch "
        f"--root {target} --source-dir {source}"
    )
    check = f"-m app.ocr_engine.russian.assets check --root {target}"
    assert fetch in calls
    assert check in calls
    assert calls.index(fetch) < calls.index(check)
    assert f"RUSSIAN_OCR_MODELS_DIR={target}" in (project / ".env").read_text(
        encoding="utf-8"
    )


def test_installer_stages_frontend_before_touching_runtime(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    fake_bin = tmp_path / "bin"
    project.mkdir()
    fake_bin.mkdir()
    shutil.copy2(INSTALLER, project / "install.sh")
    (project / "profiles" / "cpu").mkdir(parents=True)
    frontend = project / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text(
        '{"scripts":{"build":"test-build"}}',
        encoding="utf-8",
    )
    (frontend / "package-lock.json").write_text("{}", encoding="utf-8")

    fake_python = fake_bin / "python3.11"
    fake_python.write_text(
        '#!/bin/sh\nif [ "${1:-}" = "-c" ]; then echo 3.11; fi\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    fake_poetry = fake_bin / "poetry"
    fake_poetry.write_text(
        "#!/bin/sh\n"
        "profile=\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-C" ]; then profile=$2; shift 2; '
        "else shift; fi\n"
        "done\n"
        'mkdir -p "$profile/.venv/bin"\n'
        "cat > \"$profile/.venv/bin/python\" <<'PY'\n"
        "#!/bin/sh\n"
        'if [ "${1:-}" = "-c" ]; then echo test-key; fi\n'
        "PY\n"
        'chmod +x "$profile/.venv/bin/python"\n',
        encoding="utf-8",
    )
    fake_poetry.chmod(0o755)

    fake_node = fake_bin / "node"
    fake_node.write_text(
        '#!/bin/sh\nif [ "${1:-}" = "-p" ]; then echo 20; fi\n',
        encoding="utf-8",
    )
    fake_node.chmod(0o755)

    npm_log = tmp_path / "npm.log"
    fake_npm = fake_bin / "npm"
    fake_npm.write_text(
        "#!/bin/sh\n"
        'printf \'%s|%s\\n\' "$PWD" "$*" >> "$INSTALL_TEST_NPM_LOG"\n'
        'if [ "${1:-}" = "run" ] && [ "${2:-}" = "build" ]; then\n'
        '  out_dir="$PWD/../app/static/ui"\n'
        '  while [ "$#" -gt 0 ]; do\n'
        '    if [ "$1" = "--outDir" ]; then out_dir=$2; shift 2; '
        "else shift; fi\n"
        "  done\n"
        '  mkdir -p "$out_dir"\n'
        "  printf '<!doctype html>' > \"$out_dir/index.html\"\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_npm.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(project / "install.sh"),
            "--edition",
            "manual",
            "--device",
            "cpu",
            "--yes",
            "--no-start",
        ],
        cwd=project,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "INSTALL_TEST_NPM_LOG": str(npm_log),
            "OCR_MODELS": "paddle",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.index(
        "Устанавливаю зависимости веб-интерфейса"
    ) < result.stdout.index("OCR API не запущен")
    assert result.stdout.index("OCR API не запущен") < result.stdout.index(
        "Устанавливаю Python-профиль cpu"
    )
    assert npm_log.read_text(encoding="utf-8").splitlines() == [
        f"{frontend}|ci --include=dev",
        (
            f"{frontend}|run build -- --outDir "
            f"{project / '.runtime' / 'frontend-next'}"
        ),
    ]
    assert (project / "app" / "static" / "ui" / "index.html").is_file()


def test_installer_fails_clearly_when_node_is_missing(tmp_path: Path) -> None:
    project = tmp_path / "project"
    fake_bin = tmp_path / "bin"
    project.mkdir()
    fake_bin.mkdir()
    shutil.copy2(INSTALLER, project / "install.sh")
    (project / "frontend").mkdir()
    (project / "frontend" / "package.json").write_text("{}", encoding="utf-8")
    (project / "frontend" / "package-lock.json").write_text(
        "{}", encoding="utf-8"
    )

    for name, body in {
        "python3.11": "#!/bin/sh\necho 3.11\n",
        "poetry": "#!/bin/sh\nexit 0\n",
    }.items():
        executable = fake_bin / name
        executable.write_text(body, encoding="utf-8")
        executable.chmod(0o755)

    result = subprocess.run(
        [
            "/bin/bash",
            str(project / "install.sh"),
            "--edition",
            "manual",
            "--device",
            "cpu",
            "--yes",
            "--no-start",
        ],
        cwd=project,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "OCR_MODELS": "paddle",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Не найдена обязательная команда: node" in result.stderr


def test_restart_validates_frontend_before_stopping_services(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy2(INSTALLER, project / "install.sh")

    result = subprocess.run(
        ["/bin/bash", str(project / "install.sh"), "restart"],
        cwd=project,
        env=os.environ,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Веб-интерфейс не собран" in result.stderr
    assert "OCR API не запущен" not in result.stdout
