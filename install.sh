#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/.env"
RUNTIME_DIR="$ROOT_DIR/.runtime"
RUN_DIR="$ROOT_DIR/storage/run"
LOG_DIR_DEFAULT="$ROOT_DIR/logs"
FRONTEND_DIR="$ROOT_DIR/frontend"
FRONTEND_DIST_DIR="$ROOT_DIR/app/static/ui"
FRONTEND_STAGE_DIR="$RUNTIME_DIR/frontend-next"

COMMAND="install"
COMMAND_SET=0
EDITION=""
ACCELERATOR=""
ASSUME_YES=0
DRY_RUN=0
NO_START=0

MODELS=""
RUNTIME_PROFILE=""
CORE_DEVICE=""
RUSSIAN_DEVICE=""
QWEN_DEVICE_VALUE=""
ACCELERATOR_LABEL=""
LLAMA_SERVER_BIN=""

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_BLUE=$'\033[34m'
  C_CYAN=$'\033[36m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_RED=$'\033[31m'
else
  C_RESET=""
  C_BOLD=""
  C_BLUE=""
  C_CYAN=""
  C_GREEN=""
  C_YELLOW=""
  C_RED=""
fi

banner() {
  printf '%s\n' "${C_BLUE}${C_BOLD}╭──────────────────────────────────────────────╮${C_RESET}"
  printf '%s\n' "${C_BLUE}${C_BOLD}│              OCR Service Installer           │${C_RESET}"
  printf '%s\n' "${C_BLUE}${C_BOLD}╰──────────────────────────────────────────────╯${C_RESET}"
}

info() { printf '%s\n' "${C_CYAN}›${C_RESET} $*"; }
success() { printf '%s\n' "${C_GREEN}✓${C_RESET} $*"; }
warn() { printf '%s\n' "${C_YELLOW}!${C_RESET} $*"; }
die() {
  printf '%s\n' "${C_RED}✗ $*${C_RESET}" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  ./install.sh [install] [options]
  ./install.sh start|stop|restart|status

Install options:
  --edition standard|extended|manual
  --device cpu|gpu|npu  NPU — только явная диагностическая проверка
  --yes                 Do not ask for confirmation
  --dry-run             Show the resolved plan without changing the system
  --no-start            Install and configure without starting services
  -h, --help            Show this help

Editions:
  standard  PaddleOCR, MRZ Scanner and Russian OCR
  extended  Standard edition plus Qwen3-VL 4B and 8B
  manual    Read the comma-separated OCR_MODELS list from .env

Requirements:
  Python 3.11, Poetry, Node.js 18/20/22+ with npm, and Docker Compose when services start

Examples:
  ./install.sh
  ./install.sh --edition standard --device cpu --yes
  OCR_MODELS=paddle,mrz ./install.sh --edition manual --device cpu --yes
EOF
}

parse_args() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      install|start|stop|restart|status)
        if [ "$COMMAND_SET" -eq 1 ]; then
          die "Указано больше одной команды"
        fi
        COMMAND="$1"
        COMMAND_SET=1
        shift
        ;;
      --edition)
        [ "$#" -ge 2 ] || die "После --edition требуется значение"
        EDITION="$2"
        shift 2
        ;;
      --device)
        [ "$#" -ge 2 ] || die "После --device требуется значение"
        ACCELERATOR="$2"
        shift 2
        ;;
      --yes|-y)
        ASSUME_YES=1
        shift
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --no-start)
        NO_START=1
        shift
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        die "Неизвестный аргумент: $1"
        ;;
    esac
  done
}

load_env() {
  local line key value
  [ -f "$ENV_FILE" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    if [ -z "$line" ] || [[ "$line" == \#* ]] || [[ "$line" != *=* ]]; then
      continue
    fi
    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || [ -n "${!key+x}" ]; then
      continue
    fi
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    export "$key=$value"
  done < "$ENV_FILE"
}

upsert_env() {
  local key="$1" value="$2" temporary
  export "$key=$value"
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  temporary="$(mktemp "$ROOT_DIR/.env.tmp.XXXXXX")"
  umask 077
  if [ -f "$ENV_FILE" ]; then
    awk -v key="$key" -v value="$value" '
      BEGIN { replaced = 0 }
      $0 ~ "^[[:space:]]*" key "[[:space:]]*=" {
        if (!replaced) print key "=" value
        replaced = 1
        next
      }
      { print }
      END { if (!replaced) print key "=" value }
    ' "$ENV_FILE" > "$temporary"
  else
    printf '%s=%s\n' "$key" "$value" > "$temporary"
  fi
  mv "$temporary" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
}

remove_env() {
  local key="$1" temporary
  unset "$key"
  if [ "$DRY_RUN" -eq 1 ] || [ ! -f "$ENV_FILE" ]; then
    return 0
  fi
  temporary="$(mktemp "$ROOT_DIR/.env.tmp.XXXXXX")"
  umask 077
  awk -v key="$key" '
    $0 !~ "^[[:space:]]*" key "[[:space:]]*=" { print }
  ' "$ENV_FILE" > "$temporary"
  mv "$temporary" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
}

contains_model() {
  case ",$MODELS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

normalize_models() {
  local raw="$1" item normalized="" unknown=""
  local old_ifs="$IFS"
  IFS=','
  for item in $raw; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    item="$(printf '%s' "$item" | tr '[:upper:]' '[:lower:]')"
    [ -n "$item" ] || continue
    case "$item" in
      paddle|mrz|russian-ocr|qwen3-vl-4b|qwen3-vl-8b) ;;
      *) unknown="${unknown}${unknown:+, }$item"; continue ;;
    esac
    case ",$normalized," in
      *",$item,"*) ;;
      *) normalized="${normalized}${normalized:+,}$item" ;;
    esac
  done
  IFS="$old_ifs"
  [ -z "$unknown" ] || die "OCR_MODELS содержит неизвестные модели: $unknown"
  [ -n "$normalized" ] || die "OCR_MODELS должен содержать хотя бы одну модель"
  MODELS="$normalized"
}

choose_edition() {
  if [ -n "$EDITION" ]; then
    return
  fi
  [ -t 0 ] || die "Укажи --edition для запуска без терминала"
  printf '\n%s\n' "${C_BOLD}Выбери комплектацию:${C_RESET}"
  printf '  %s1%s  Стандартная  — PaddleOCR, MRZ Scanner, Russian OCR\n' "$C_CYAN" "$C_RESET"
  printf '  %s2%s  Расширенная — стандартная + Qwen3-VL 4B/8B\n' "$C_CYAN" "$C_RESET"
  printf '  %s3%s  Ручная      — список OCR_MODELS из .env\n' "$C_CYAN" "$C_RESET"
  printf '> '
  read -r answer
  case "$answer" in
    1) EDITION="standard" ;;
    2) EDITION="extended" ;;
    3) EDITION="manual" ;;
    *) die "Неизвестный вариант комплектации" ;;
  esac
}

resolve_edition() {
  EDITION="$(printf '%s' "$EDITION" | tr '[:upper:]' '[:lower:]')"
  case "$EDITION" in
    standard)
      MODELS="paddle,mrz,russian-ocr"
      ;;
    extended|advanced)
      EDITION="extended"
      MODELS="paddle,mrz,russian-ocr,qwen3-vl-4b,qwen3-vl-8b"
      ;;
    manual|custom)
      EDITION="manual"
      [ -n "${OCR_MODELS:-}" ] || die "Для ручной комплектации задай OCR_MODELS в .env"
      normalize_models "$OCR_MODELS"
      ;;
    *)
      die "Комплектация должна быть standard, extended или manual"
      ;;
  esac
}

choose_accelerator() {
  if [ -n "$ACCELERATOR" ]; then
    return
  fi
  [ -t 0 ] || die "Укажи --device для запуска без терминала"
  printf '\n%s\n' "${C_BOLD}Выбери режим вычислений:${C_RESET}"
  printf '  %s1%s  CPU\n' "$C_CYAN" "$C_RESET"
  printf '  %s2%s  GPU\n' "$C_CYAN" "$C_RESET"
  printf '> '
  read -r answer
  case "$answer" in
    1) ACCELERATOR="cpu" ;;
    2) ACCELERATOR="gpu" ;;
    *) die "Неизвестный режим вычислений" ;;
  esac
}

detect_npu() {
  local os arch compatible driver_path driver_name pci_text
  os="$(uname -s)"
  arch="$(uname -m)"
  DETECTED_NPU=""
  DETECTED_NPU_DETAILS=""

  if [ "$os" = "Darwin" ] && [ "$arch" = "arm64" ]; then
    if command -v ioreg >/dev/null 2>&1 && ioreg -r -c H11ANEIn 2>/dev/null | grep -q H11ANEIn; then
      DETECTED_NPU="apple-ane"
      DETECTED_NPU_DETAILS="Apple Neural Engine"
    else
      DETECTED_NPU="apple-ane"
      DETECTED_NPU_DETAILS="Apple Silicon Neural Engine (определён по платформе)"
    fi
    return
  fi

  if [ "$os" != "Linux" ]; then
    return
  fi

  if command -v npu-smi >/dev/null 2>&1 || [ -e /dev/davinci_manager ]; then
    DETECTED_NPU="huawei-ascend"
    DETECTED_NPU_DETAILS="Huawei Ascend"
    return
  fi
  if compgen -G '/dev/rknpu*' >/dev/null 2>&1; then
    DETECTED_NPU="rockchip-rknn"
    DETECTED_NPU_DETAILS="Rockchip RKNN"
    return
  fi

  compatible=""
  if [ -r /proc/device-tree/compatible ]; then
    compatible="$(tr '\0' ' ' < /proc/device-tree/compatible 2>/dev/null || true)"
  fi
  case "$compatible" in
    *rockchip*)
      DETECTED_NPU="rockchip-rknn"
      DETECTED_NPU_DETAILS="Rockchip RKNN"
      return
      ;;
    *qcom*|*qualcomm*)
      DETECTED_NPU="qualcomm-hexagon"
      DETECTED_NPU_DETAILS="Qualcomm Hexagon"
      return
      ;;
  esac

  for driver_path in /sys/class/accel/accel*/device/driver; do
    [ -e "$driver_path" ] || continue
    driver_name="$(basename "$(readlink -f "$driver_path" 2>/dev/null || printf '%s' "$driver_path")")"
    case "$driver_name" in
      intel_vpu*)
        DETECTED_NPU="intel-npu"
        DETECTED_NPU_DETAILS="Intel NPU ($driver_name)"
        return
        ;;
      amdxdna*)
        DETECTED_NPU="amd-xdna"
        DETECTED_NPU_DETAILS="AMD XDNA ($driver_name)"
        return
        ;;
    esac
  done

  if command -v lspci >/dev/null 2>&1; then
    pci_text="$(lspci -nn 2>/dev/null || true)"
    if printf '%s' "$pci_text" | grep -Eqi 'Intel.*(NPU|Neural Processing|AI Boost)'; then
      DETECTED_NPU="intel-npu"
      DETECTED_NPU_DETAILS="Intel NPU"
      return
    fi
    if printf '%s' "$pci_text" | grep -Eqi 'AMD.*(XDNA|IPU|Neural Processing)'; then
      DETECTED_NPU="amd-xdna"
      DETECTED_NPU_DETAILS="AMD XDNA"
      return
    fi
  fi
}

resolve_accelerator() {
  local os arch
  ACCELERATOR="$(printf '%s' "$ACCELERATOR" | tr '[:upper:]' '[:lower:]')"
  os="$(uname -s)"
  arch="$(uname -m)"
  case "$ACCELERATOR" in
    cpu)
      case "$os/$arch" in
        Darwin/arm64|Darwin/x86_64|Linux/x86_64) ;;
        *)
          die "Для платформы $os/$arch нет проверенного CPU-профиля. Установка не начата"
          ;;
      esac
      RUNTIME_PROFILE="cpu"
      CORE_DEVICE="cpu"
      RUSSIAN_DEVICE="cpu"
      QWEN_DEVICE_VALUE="cpu"
      ACCELERATOR_LABEL="CPU"
      ;;
    gpu)
      if [ "$os" = "Linux" ] && [ "$arch" = "x86_64" ]; then
        command -v nvidia-smi >/dev/null 2>&1 || die "NVIDIA GPU не обнаружен: команда nvidia-smi отсутствует"
        nvidia-smi -L >/dev/null 2>&1 || die "NVIDIA GPU обнаружен, но драйвер недоступен"
        RUNTIME_PROFILE="gpu"
        CORE_DEVICE="gpu"
        RUSSIAN_DEVICE="gpu"
        QWEN_DEVICE_VALUE="auto"
        ACCELERATOR_LABEL="NVIDIA CUDA; MRZ Scanner остаётся на CPU"
      elif [ "$os" = "Darwin" ] && [ "$arch" = "arm64" ]; then
        if ! contains_model qwen3-vl-4b && ! contains_model qwen3-vl-8b; then
          die "В выбранной комплектации нет модели с поддержкой Apple Metal"
        fi
        RUNTIME_PROFILE="cpu"
        CORE_DEVICE="cpu"
        RUSSIAN_DEVICE="cpu"
        QWEN_DEVICE_VALUE="auto"
        ACCELERATOR_LABEL="Apple Metal для Qwen; остальные модели работают на CPU"
      else
        die "Для этой платформы нет проверенного GPU-профиля: $os/$arch"
      fi
      ;;
    npu)
      detect_npu
      if [ -z "$DETECTED_NPU" ]; then
        die "NPU не обнаружен. Установка не начата"
      fi
      die "Обнаружен $DETECTED_NPU_DETAILS, но модели $MODELS не имеют проверенного профиля $DETECTED_NPU. Установка не начата"
      ;;
    *)
      die "Режим должен быть cpu, gpu или npu"
      ;;
  esac
}

model_device_summary() {
  if contains_model paddle; then
    if [ "$CORE_DEVICE" = "gpu" ]; then
      printf '    PaddleOCR       NVIDIA CUDA\n'
    else
      printf '    PaddleOCR       CPU\n'
    fi
  fi
  if contains_model mrz; then
    printf '    MRZ Scanner     CPU\n'
  fi
  if contains_model russian-ocr; then
    if [ "$RUSSIAN_DEVICE" = "gpu" ]; then
      printf '    Russian OCR     NVIDIA CUDA\n'
    else
      printf '    Russian OCR     CPU\n'
    fi
  fi
  if contains_model qwen3-vl-4b || contains_model qwen3-vl-8b; then
    if [ "$QWEN_DEVICE_VALUE" = "cpu" ]; then
      printf '    Qwen3-VL        CPU\n'
    elif [ "$(uname -s)" = "Darwin" ]; then
      printf '    Qwen3-VL        Apple Metal\n'
    else
      printf '    Qwen3-VL        NVIDIA CUDA\n'
    fi
  fi
}

show_plan() {
  printf '\n%s\n' "${C_BOLD}План установки${C_RESET}"
  printf '  Комплектация:  %s\n' "$EDITION"
  printf '  Модели:        %s\n' "$MODELS"
  printf '  Режим:         %s\n' "$ACCELERATOR_LABEL"
  printf '  Профиль:       profiles/%s\n' "$RUNTIME_PROFILE"
  printf '  Распределение:\n'
  model_device_summary
  printf '\n'
}

confirm_plan() {
  [ "$ASSUME_YES" -eq 1 ] && return 0
  [ -t 0 ] || die "Для запуска без терминала добавь --yes"
  printf 'Продолжить установку? [y/N] '
  read -r answer
  case "$answer" in
    y|Y|yes|YES|да|Да|ДА) ;;
    *) info "Установка отменена"; exit 0 ;;
  esac
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Не найдена обязательная команда: $1"
}

python_command() {
  local candidate version
  for candidate in python3.11 python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    version="$($candidate -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [ "$version" = "3.11" ]; then
      command -v "$candidate"
      return
    fi
  done
  return 1
}

resolve_llama_server() {
  local configured="${VLM_LLAMA_SERVER_BIN:-}" resolved
  if [ -n "$configured" ]; then
    if [ -x "$configured" ]; then
      LLAMA_SERVER_BIN="$configured"
      return
    fi
    resolved="$(command -v "$configured" 2>/dev/null || true)"
    [ -n "$resolved" ] || die "VLM_LLAMA_SERVER_BIN недоступен: $configured"
    LLAMA_SERVER_BIN="$resolved"
    return
  fi
  if command -v llama-server >/dev/null 2>&1; then
    LLAMA_SERVER_BIN="$(command -v llama-server)"
    return
  fi
  if command -v brew >/dev/null 2>&1; then
    LLAMA_SERVER_BIN="brew-install"
    return
  fi
  die "Для Qwen нужен llama-server. Установи llama.cpp или задай VLM_LLAMA_SERVER_BIN"
}

verify_llama_acceleration() {
  local devices os
  if [ "$ACCELERATOR" != "gpu" ] || [ "$LLAMA_SERVER_BIN" = "brew-install" ]; then
    return
  fi
  devices="$("$LLAMA_SERVER_BIN" --list-devices 2>&1 || true)"
  os="$(uname -s)"
  if [ "$os" = "Darwin" ]; then
    printf '%s' "$devices" | grep -qi metal \
      || die "llama-server не видит Metal; GPU-профиль Qwen не может быть подтверждён"
  else
    printf '%s' "$devices" | grep -qi cuda \
      || die "llama-server не видит CUDA; установи CUDA-сборку llama.cpp"
  fi
}

validate_runtime_path() {
  local path="$1"
  if [ -e "$path" ] && [ ! -L "$path" ]; then
    die "$path уже существует и не является символической ссылкой; перенеси этот каталог перед установкой"
  fi
}

preflight() {
  local node_major
  require_command poetry
  PYTHON_BIN="$(python_command)" || die "Требуется Python 3.11"
  [ -f "$FRONTEND_DIR/package.json" ] \
    || die "Не найден frontend/package.json; веб-интерфейс не может быть собран"
  [ -f "$FRONTEND_DIR/package-lock.json" ] \
    || die "Не найден frontend/package-lock.json; установка должна быть воспроизводимой"
  require_command node
  require_command npm
  node_major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null)" \
    || die "Не удалось определить версию Node.js"
  [[ "$node_major" =~ ^[0-9]+$ ]] \
    || die "Некорректная версия Node.js: $node_major"
  if [ "$node_major" -ne 18 ] && [ "$node_major" -ne 20 ] && [ "$node_major" -lt 22 ]; then
    die "Требуется Node.js 18, 20 или 22+; обнаружена версия $(node --version)"
  fi
  if [ "$NO_START" -eq 0 ]; then
    require_command docker
    docker compose version >/dev/null 2>&1 || die "Требуется Docker Compose v2"
    require_command curl
  fi
  if contains_model qwen3-vl-4b || contains_model qwen3-vl-8b; then
    resolve_llama_server
    verify_llama_acceleration
  fi
  validate_runtime_path "$RUNTIME_DIR/api"
  validate_runtime_path "$RUNTIME_DIR/vlm"
}

install_frontend() {
  info "Устанавливаю зависимости веб-интерфейса"
  (cd "$FRONTEND_DIR" && npm ci --include=dev)

  info "Собираю веб-интерфейс"
  rm -rf "$FRONTEND_STAGE_DIR"
  (cd "$FRONTEND_DIR" && npm run build -- --outDir "$FRONTEND_STAGE_DIR")
  [ -f "$FRONTEND_STAGE_DIR/index.html" ] \
    || die "Frontend build завершился без index.html"
  success "Веб-интерфейс собран и готов к активации"
}

activate_frontend() {
  local backup_dir="$RUNTIME_DIR/frontend-previous"
  rm -rf "$backup_dir"
  mkdir -p "$(dirname "$FRONTEND_DIST_DIR")"
  if [ -d "$FRONTEND_DIST_DIR" ]; then
    mv "$FRONTEND_DIST_DIR" "$backup_dir"
  fi
  if ! mv "$FRONTEND_STAGE_DIR" "$FRONTEND_DIST_DIR"; then
    if [ -d "$backup_dir" ]; then
      mv "$backup_dir" "$FRONTEND_DIST_DIR"
    fi
    die "Не удалось активировать новую сборку веб-интерфейса"
  fi
  rm -rf "$backup_dir"
  success "Веб-интерфейс активирован"
}

require_frontend_build() {
  [ -f "$FRONTEND_DIST_DIR/index.html" ] \
    || die "Веб-интерфейс не собран; запусти ./install.sh"
}

install_profile() {
  local profile="$1"
  info "Устанавливаю Python-профиль $profile"
  POETRY_VIRTUALENVS_IN_PROJECT=true poetry -C "$ROOT_DIR/profiles/$profile" install --only main --no-interaction
  [ -x "$ROOT_DIR/profiles/$profile/.venv/bin/python" ] || die "Poetry не создал profiles/$profile/.venv"
}

install_llama_server_if_needed() {
  if [ "$LLAMA_SERVER_BIN" != "brew-install" ]; then
    return
  fi
  info "Устанавливаю llama.cpp через Homebrew"
  brew install llama.cpp
  LLAMA_SERVER_BIN="$(command -v llama-server 2>/dev/null || true)"
  [ -x "$LLAMA_SERVER_BIN" ] || die "Homebrew не установил llama-server"
  verify_llama_acceleration
}

install_qwen() {
  if ! contains_model qwen3-vl-4b && ! contains_model qwen3-vl-8b; then
    return
  fi
  export HF_HOME="$ROOT_DIR/storage/vlm/huggingface"
  install_llama_server_if_needed
  install_profile "vlm"
  if contains_model qwen3-vl-4b && contains_model qwen3-vl-8b; then
    info "Загружаю Qwen3-VL 4B и 8B"
    "$ROOT_DIR/profiles/vlm/.venv/bin/python" "$ROOT_DIR/scripts/models/download-vlm-models.py" --model all
  elif contains_model qwen3-vl-4b; then
    info "Загружаю Qwen3-VL 4B"
    "$ROOT_DIR/profiles/vlm/.venv/bin/python" "$ROOT_DIR/scripts/models/download-vlm-models.py" --model qwen3-vl-4b
  else
    info "Загружаю Qwen3-VL 8B"
    "$ROOT_DIR/profiles/vlm/.venv/bin/python" "$ROOT_DIR/scripts/models/download-vlm-models.py" --model qwen3-vl-8b
  fi
}

verify_profile() {
  local python="$ROOT_DIR/profiles/$RUNTIME_PROFILE/.venv/bin/python"
  info "Проверяю установленные OCR-зависимости"
  "$python" - <<'PY'
import importlib.util

import cv2
import onnxruntime
import paddle
import paddleocr

if importlib.util.find_spec("mrzscanner") is None:
    raise SystemExit("Пакет mrzscanner не установлен")
PY
  if [ "$CORE_DEVICE" = "gpu" ]; then
    "$python" - <<'PY'
import onnxruntime
import paddle

if not paddle.device.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
    raise SystemExit("PaddleOCR не видит CUDA GPU")
if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
    raise SystemExit("ONNX Runtime не видит CUDAExecutionProvider")
PY
  fi
}

prepare_mrz_assets() {
  local python="$ROOT_DIR/profiles/$RUNTIME_PROFILE/.venv/bin/python"
  contains_model mrz || return 0
  info "Подготавливаю локальные веса MRZ Scanner"
  "$python" - <<'PY'
from importlib.util import find_spec
from pathlib import Path
import shutil
import subprocess

spec = find_spec("capybara")
if spec is None or spec.origin is None:
    raise SystemExit("Пакет capybara не установлен")
font_target = Path(spec.origin).parent / "vision" / "visualization" / "NotoSansMonoCJKtc-VF.ttf"
if not font_target.exists():
    candidates = [
        Path("/System/Library/Fonts/SFNSMono.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf"),
    ]
    if shutil.which("fc-match"):
        match = subprocess.run(
            ["fc-match", "monospace", "-f", "%{file}"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if match:
            candidates.append(Path(match))
    source = next((path for path in candidates if path.is_file()), None)
    if source is not None:
        font_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, font_target)

from app.ocr_engine.mrz.factory import create_mrz_scanner

create_mrz_scanner()
PY
}

prepare_russian_ocr_assets() {
  local python="$ROOT_DIR/profiles/$RUNTIME_PROFILE/.venv/bin/python"
  local target source
  contains_model russian-ocr || return 0

  target="${RUSSIAN_OCR_MODELS_DIR:-}"
  if [ -z "$target" ]; then
    target="$(cd "$ROOT_DIR" && "$python" -c \
      'from app.ocr_engine.russian.assets import models_dir; print(models_dir())')"
  fi
  if [[ "$target" != /* ]]; then
    target="$ROOT_DIR/$target"
  fi
  RUSSIAN_OCR_MODELS_DIR="$target"
  export RUSSIAN_OCR_MODELS_DIR

  source="${RUSSIAN_OCR_MODELS_SOURCE_DIR:-}"
  if [ -n "$source" ] && [[ "$source" != /* ]]; then
    source="$ROOT_DIR/$source"
  fi

  if [ -n "$source" ]; then
    info "Копирую локальные веса Russian OCR"
    (
      cd "$ROOT_DIR"
      "$python" -m app.ocr_engine.russian.assets fetch \
        --root "$target" \
        --source-dir "$source"
    )
  else
    info "Загружаю веса Russian OCR"
    (
      cd "$ROOT_DIR"
      "$python" -m app.ocr_engine.russian.assets fetch --root "$target"
    )
  fi

  info "Проверяю веса Russian OCR"
  (
    cd "$ROOT_DIR"
    "$python" -m app.ocr_engine.russian.assets check --root "$target"
  )
}

activate_runtime() {
  validate_runtime_path "$RUNTIME_DIR/api"
  validate_runtime_path "$RUNTIME_DIR/vlm"
  mkdir -p "$RUNTIME_DIR"
  ln -sfn "$ROOT_DIR/profiles/$RUNTIME_PROFILE/.venv" "$RUNTIME_DIR/api"
  if contains_model qwen3-vl-4b || contains_model qwen3-vl-8b; then
    ln -sfn "$ROOT_DIR/profiles/vlm/.venv" "$RUNTIME_DIR/vlm"
  else
    rm -f "$RUNTIME_DIR/vlm"
  fi
}

ensure_qr_key() {
  local python="$1" key_id generated
  if [ -n "${QR_ENCRYPTION_KEYS:-}" ]; then
    return
  fi
  key_id="${QR_ACTIVE_KEY_ID:-primary}"
  [[ "$key_id" =~ ^[A-Za-z0-9._-]{1,64}$ ]] || die "QR_ACTIVE_KEY_ID содержит недопустимые символы"
  generated="$($python -c 'import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("="))')"
  upsert_env QR_ACTIVE_KEY_ID "$key_id"
  upsert_env QR_ENCRYPTION_KEYS "$key_id:$generated"
}

write_configuration() {
  remove_env OCR_EDITION
  remove_env OCR_ACCELERATOR
  upsert_env OCR_MODELS "$MODELS"
  upsert_env OCR_RUNTIME_PROFILE "$RUNTIME_PROFILE"
  upsert_env OCR_DEVICE "$CORE_DEVICE"
  upsert_env RUSSIAN_OCR_DEVICE "$RUSSIAN_DEVICE"
  remove_env RUSSIAN_OCR_MODEL_FORMAT
  if contains_model russian-ocr; then
    upsert_env RUSSIAN_OCR_MODELS_DIR "$RUSSIAN_OCR_MODELS_DIR"
  else
    remove_env RUSSIAN_OCR_MODELS_DIR
    remove_env RUSSIAN_OCR_MODELS_SOURCE_DIR
  fi
  if contains_model qwen3-vl-4b || contains_model qwen3-vl-8b; then
    upsert_env QWEN_DEVICE "$QWEN_DEVICE_VALUE"
    upsert_env HF_HOME "$ROOT_DIR/storage/vlm/huggingface"
    if [ -n "$LLAMA_SERVER_BIN" ] && [ "$LLAMA_SERVER_BIN" != "brew-install" ]; then
      upsert_env VLM_LLAMA_SERVER_BIN "$LLAMA_SERVER_BIN"
    fi
  else
    remove_env QWEN_DEVICE
    remove_env HF_HOME
    remove_env VLM_LLAMA_SERVER_BIN
  fi
  ensure_qr_key "$ROOT_DIR/profiles/$RUNTIME_PROFILE/.venv/bin/python"
}

pid_is_running() {
  local pid_file="$1" pid
  [ -f "$pid_file" ] || return 1
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

managed_pid_is_running() {
  local pid_file="$1" marker="$2" pid command_line
  pid_is_running "$pid_file" || return 1
  pid="$(cat "$pid_file")"
  command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  case "$command_line" in
    *"$marker"*) return 0 ;;
    *) return 1 ;;
  esac
}

start_process() {
  local name="$1" pid_file="$2" log_file="$3" marker="$4"
  shift 4
  if managed_pid_is_running "$pid_file" "$marker"; then
    info "$name уже запущен (PID $(cat "$pid_file"))"
    return
  fi
  if pid_is_running "$pid_file"; then
    die "PID $(cat "$pid_file") из $pid_file принадлежит другому процессу"
  fi
  rm -f "$pid_file"
  mkdir -p "$RUN_DIR" "$(dirname "$log_file")"
  (
    cd "$ROOT_DIR"
    nohup "$@" >> "$log_file" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$pid_file"
  )
  sleep 1
  if ! managed_pid_is_running "$pid_file" "$marker"; then
    rm -f "$pid_file"
    die "$name не запустился; смотри $log_file"
  fi
  success "$name запущен (PID $(cat "$pid_file"))"
}

wait_http() {
  local name="$1" url="$2" timeout="$3" started now
  started="$(date +%s)"
  while true; do
    if curl --silent --show-error --fail --max-time 2 "$url" >/dev/null 2>&1; then
      success "$name готов: $url"
      return
    fi
    now="$(date +%s)"
    if [ $((now - started)) -ge "$timeout" ]; then
      printf '%s\n' "${C_RED}✗ $name не стал готов за ${timeout}с${C_RESET}" >&2
      return 1
    fi
    sleep 1
  done
}

runtime_profile_from_env() {
  RUNTIME_PROFILE="${OCR_RUNTIME_PROFILE:-${OCR_DEVICE:-cpu}}"
  case "$RUNTIME_PROFILE" in
    cpu|gpu) ;;
    *) die "Некорректный OCR_RUNTIME_PROFILE: $RUNTIME_PROFILE" ;;
  esac
  MODELS="${OCR_MODELS:-paddle,mrz,russian-ocr,qwen3-vl-4b,qwen3-vl-8b}"
  normalize_models "$MODELS"
}

ensure_runtime_links() {
  local api_env="$ROOT_DIR/profiles/$RUNTIME_PROFILE/.venv"
  [ -x "$api_env/bin/python" ] || die "Профиль $RUNTIME_PROFILE не установлен; запусти ./install.sh"
  validate_runtime_path "$RUNTIME_DIR/api"
  validate_runtime_path "$RUNTIME_DIR/vlm"
  mkdir -p "$RUNTIME_DIR"
  ln -sfn "$api_env" "$RUNTIME_DIR/api"
  if contains_model qwen3-vl-4b || contains_model qwen3-vl-8b; then
    [ -x "$ROOT_DIR/profiles/vlm/.venv/bin/python" ] || die "VLM-профиль не установлен; запусти ./install.sh"
    ln -sfn "$ROOT_DIR/profiles/vlm/.venv" "$RUNTIME_DIR/vlm"
  fi
}

start_stack() {
  local log_dir="${LOG_DIR:-$LOG_DIR_DEFAULT}" health_host
  if [[ "$log_dir" != /* ]]; then
    log_dir="$ROOT_DIR/$log_dir"
  fi
  runtime_profile_from_env
  require_frontend_build
  ensure_runtime_links
  require_command docker
  docker compose version >/dev/null 2>&1 || die "Требуется Docker Compose v2"
  require_command curl
  info "Запускаю PostgreSQL и RabbitMQ"
  (cd "$ROOT_DIR" && docker compose up -d --wait --remove-orphans postgres rabbitmq)
  info "Применяю миграции"
  (cd "$ROOT_DIR" && "$RUNTIME_DIR/api/bin/alembic" -c alembic.ini upgrade head)
  if contains_model qwen3-vl-4b || contains_model qwen3-vl-8b; then
    start_process "Qwen VLM" "$RUN_DIR/vlm.pid" "$log_dir/vlm.log" "app.modules.vlm.server" \
      "$RUNTIME_DIR/vlm/bin/python" -m app.modules.vlm.server
    if ! wait_http "Qwen VLM" "${VLM_OCR_URL:-http://127.0.0.1:8004}/health" "${VLM_LLAMA_STARTUP_TIMEOUT_SECONDS:-300}"; then
      stop_one "Qwen VLM" "$RUN_DIR/vlm.pid" "app.modules.vlm.server"
      die "Qwen VLM не прошёл проверку готовности"
    fi
  fi
  start_process "OCR API" "$RUN_DIR/api.pid" "$log_dir/api.log" "app.main" \
    "$RUNTIME_DIR/api/bin/python" -m app.main
  health_host="${API_HOST:-127.0.0.1}"
  case "$health_host" in
    0.0.0.0|::|\[::\]) health_host="127.0.0.1" ;;
  esac
  if ! wait_http "OCR API" "http://${health_host}:${API_PORT:-8000}/health" "${OCR_INSTALL_STARTUP_TIMEOUT_SECONDS:-360}"; then
    stop_one "OCR API" "$RUN_DIR/api.pid" "app.main"
    stop_one "Qwen VLM" "$RUN_DIR/vlm.pid" "app.modules.vlm.server"
    die "OCR API не прошёл проверку готовности"
  fi
  printf '\n'
  success "OCR Service запущен"
  printf '  API:      http://%s:%s\n' "${API_HOST:-127.0.0.1}" "${API_PORT:-8000}"
  printf '  UI:       http://%s:%s/ui/\n' "${API_HOST:-127.0.0.1}" "${API_PORT:-8000}"
  printf '  Swagger:  http://%s:%s/docs\n' "${API_HOST:-127.0.0.1}" "${API_PORT:-8000}"
  printf '  Логи:     %s\n' "$log_dir"
}

stop_one() {
  local name="$1" pid_file="$2" marker="$3" pid count
  if ! pid_is_running "$pid_file"; then
    rm -f "$pid_file"
    info "$name не запущен"
    return
  fi
  pid="$(cat "$pid_file")"
  managed_pid_is_running "$pid_file" "$marker" \
    || die "PID $pid из $pid_file принадлежит другому процессу"
  kill -TERM "$pid" 2>/dev/null || true
  count=0
  while kill -0 "$pid" 2>/dev/null && [ "$count" -lt 30 ]; do
    sleep 1
    count=$((count + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    warn "$name не завершился мягко; отправляю KILL"
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
  success "$name остановлен"
}

stop_stack() {
  stop_one "OCR API" "$RUN_DIR/api.pid" "app.main"
  stop_one "Qwen VLM" "$RUN_DIR/vlm.pid" "app.modules.vlm.server"
}

status_one() {
  local name="$1" pid_file="$2" marker="$3"
  if managed_pid_is_running "$pid_file" "$marker"; then
    printf '  %-12s %s (PID %s)\n' "$name" "${C_GREEN}running${C_RESET}" "$(cat "$pid_file")"
  else
    printf '  %-12s %s\n' "$name" "${C_YELLOW}stopped${C_RESET}"
  fi
}

status_stack() {
  printf '%s\n' "${C_BOLD}OCR Service status${C_RESET}"
  status_one "API" "$RUN_DIR/api.pid" "app.main"
  status_one "Qwen VLM" "$RUN_DIR/vlm.pid" "app.modules.vlm.server"
  if command -v docker >/dev/null 2>&1; then
    printf '\n'
    if docker info >/dev/null 2>&1; then
      (cd "$ROOT_DIR" && docker compose ps postgres rabbitmq) || true
    else
      warn "Docker daemon недоступен"
    fi
  fi
}

install_stack() {
  choose_edition
  resolve_edition
  choose_accelerator
  resolve_accelerator
  show_plan
  confirm_plan
  if [ "$DRY_RUN" -eq 1 ]; then
    success "Проверка завершена: система не изменена"
    return
  fi
  preflight
  # Build the UI before downtime and keep the current build intact until
  # every dependency has been prepared successfully.
  install_frontend
  # The active API can point at the profile that Poetry is about to update.
  # Stop it before changing that Python environment.
  stop_stack
  install_profile "$RUNTIME_PROFILE"
  prepare_russian_ocr_assets
  verify_profile
  prepare_mrz_assets
  install_qwen
  activate_frontend
  activate_runtime
  write_configuration
  success "Установка и конфигурация завершены"
  if [ "$NO_START" -eq 0 ]; then
    load_env
    start_stack
  fi
}

main() {
  parse_args "$@"
  load_env
  case "$COMMAND" in
    install)
      banner
      install_stack
      ;;
    start)
      banner
      start_stack
      ;;
    stop)
      stop_stack
      ;;
    restart)
      load_env
      require_frontend_build
      stop_stack
      start_stack
      ;;
    status)
      status_stack
      ;;
  esac
}

main "$@"
