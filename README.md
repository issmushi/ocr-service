# OCR Service

Установка и запуск OCR-сервиса с веб-интерфейсом.
Документация API, примеры запросов и тестирование доступны в [UI](http://127.0.0.1:8000/ui/).

[Установка](#установка) · [Комплектации](#комплектации) · [Управление](#управление) · [Адреса](#адреса)

## Требования

| Компонент | Требование |
| --- | --- |
| ОС | Linux x86_64 или macOS на Intel / Apple Silicon |
| Python | **3.11** и Poetry |
| Веб-интерфейс | Node.js **18, 20 или 22+**, npm |
| Запуск сервиса | Запущенный Docker и Docker Compose **v2**, `curl` |
| Загрузка | Интернет для зависимостей, образов и моделей |

Для Qwen дополнительно нужен `llama-server` из llama.cpp. Укажите его в `PATH`
или задайте `VLM_LLAMA_SERVER_BIN` в `.env`. Если доступен Homebrew,
установщик может установить llama.cpp самостоятельно.

## Установка

Все команды выполняются из корня репозитория.

```bash
./install.sh
```

Скрипт предложит комплектацию и CPU/GPU, покажет план и запросит подтверждение.
Он установит Python-зависимости, соберёт UI, подготовит модели, сохранит настройки
и ключ QR в `.env`, запустит PostgreSQL и RabbitMQ, применит миграции и запустит API.

**Быстрый запуск на CPU без диалога:**

```bash
./install.sh --edition standard --device cpu --yes
```

После запуска откройте **[http://127.0.0.1:8000/ui/](http://127.0.0.1:8000/ui/)**.
UI раздаётся самим API; отдельный сервер для него не нужен.

### Комплектации

| `--edition` | Модели |
| --- | --- |
| `standard` | PaddleOCR, MRZ Scanner, Russian OCR |
| `extended` | Стандартная комплектация + Qwen3-VL 4B и 8B |
| `manual` | Список `OCR_MODELS` из `.env` или переменной окружения |

```bash
# Все модели с GPU
./install.sh --edition extended --device gpu --yes

# Только выбранные модели
OCR_MODELS=paddle,mrz ./install.sh --edition manual --device cpu --yes
```

Допустимые значения `OCR_MODELS`: `paddle`, `mrz`, `russian-ocr`,
`qwen3-vl-4b`, `qwen3-vl-8b` — через запятую.

### CPU и GPU

- **CPU** — Linux x86_64 и macOS на Intel / Apple Silicon.
- **GPU на Linux** — NVIDIA с рабочим драйвером (`nvidia-smi`).
  Для Qwen требуется CUDA-сборка `llama-server`. MRZ Scanner остаётся на CPU.
- **GPU на Apple Silicon** — Metal только для Qwen; остальные модели работают
  на CPU. Выберите `extended` или `manual` с Qwen. Для `standard` используйте CPU.

NPU пока не поддерживается.

### Настройки установки

| Флаг | Действие |
| --- | --- |
| `--yes` | Подтвердить выбранную комплектацию и устройство без диалога |
| `--dry-run` | Показать план без изменения файлов и установки зависимостей |
| `--no-start` | Установить и настроить сервис без запуска |
| `--help` | Показать справку |

```bash
./install.sh --edition standard --device cpu --dry-run --yes
```

Для собственных настроек создайте `.env` на основе `.env.template` **до первой
установки**. В существующей `.env` меняйте только нужные параметры.
Адрес API задаётся через `API_HOST` и `API_PORT`; по умолчанию это
`127.0.0.1:8000`. Сохраните созданный ключ QR: он нужен для чтения ранее выданных QR.

Веса Russian OCR загружаются в `storage/models/russian-docs-ocr/1.0.1/models`
и проверяются по SHA-256. Путь можно изменить через `RUSSIAN_OCR_MODELS_DIR`.
Чтобы взять эти веса из локального каталога, задайте `RUSSIAN_OCR_MODELS_SOURCE_DIR`.

## Управление

| Команда | Действие |
| --- | --- |
| `./install.sh start` | Запустить инфраструктуру и сервис |
| `./install.sh status` | Показать состояние процессов и контейнеров |
| `./install.sh restart` | Перезапустить сервис |
| `./install.sh stop` | Остановить API и Qwen VLM |

`stop` оставляет PostgreSQL и RabbitMQ работающими. Чтобы остановить и их:

```bash
docker compose stop postgres rabbitmq
```

Логи по умолчанию находятся в `logs/api.log` и, при использовании Qwen,
`logs/vlm.log`. Каталог задаётся через `LOG_DIR` в `.env`.

## Адреса

Для стандартного адреса `127.0.0.1:8000`:

| Раздел | Ссылка |
| --- | --- |
| Интерфейс и тестирование | [Открыть UI](http://127.0.0.1:8000/ui/) |
| Документация API | [Открыть документацию](http://127.0.0.1:8000/ui/docs) |
| Swagger | [Открыть Swagger](http://127.0.0.1:8000/docs) |
| Проверка состояния | [Открыть health](http://127.0.0.1:8000/health) |

<details>
<summary>Локальная разработка интерфейса</summary>

При запущенном API:

```bash
cd frontend
npm ci
npm run dev
```

Откройте [http://127.0.0.1:5173/ui/](http://127.0.0.1:5173/ui/).
Запросы `/v1` и `/health` проксируются на `http://127.0.0.1:8000`;
другой адрес задаётся через `VITE_API_PROXY_TARGET`.
Команда `npm run build` обновляет сборку в `app/static/ui`.

</details>
