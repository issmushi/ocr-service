# Интеграция с OCR выписок ЕГРН

OCR Service читает 19 полей первой страницы выписки Госуслуг из ЕГРН: 17 значений из двух таблиц, номер заявления и дату актуальности. Для немедленного ответа используйте `POST /v1/ocr/egrn`. Для фоновой обработки отправьте тот же файл в `POST /v1/tasks/egrn` и сохраните полученный `task_id`. Формат готового результата одинаков.

В Swagger эти операции находятся в группах **Russia Sync** и **Russia Async**. Группа **Tasks** содержит только получение статуса, получение результата и отмену задачи.

Базовый адрес сервера в локальной сети — `http://192.168.3.152:8100`. Клиентскому приложению нужен сетевой доступ к этому адресу. На этом сервере ключ API или заголовок авторизации не требуется. Адрес удобно вынести в настройку приложения. [Swagger](http://192.168.3.152:8100/docs) · [Проверить скан вручную](http://192.168.3.152:8100/ui/playground/egrn) · [Проверить доступность](http://192.168.3.152:8100/health)

## Что отправлять

Один файл на запрос. Передавайте светлый скан **первой страницы** с шапкой «Сведения из Единого государственного реестра недвижимости», верхней таблицей из двух строк и таблицей характеристик из 15 строк. Обработчик сам обрежет пустые поля и выровняет небольшой наклон. Если источник — PDF, отрисуйте первую страницу в PNG или JPEG перед отправкой.

Оба маршрута принимают `multipart/form-data`. При формировании формы в браузере или HTTP-клиенте дайте библиотеке поставить заголовок `Content-Type` с границей частей.

| Поле | Обязательно | Значение |
| --- | --- | --- |
| `image` | Да | Файл с именем и расширением JPG/JPEG, PNG, BMP, WebP или одностраничный TIF/TIFF. |
| `ocr_model` | Нет | `paddle-cyrillic` (по умолчанию) или `paddle` (eslav). Обе модели работают на CPU. |
| `fields` | Нет | Список нужных полей из таблицы `data` ниже (кроме `meta`). По умолчанию `all` — все 19 полей. |

Передавайте несколько одноимённых `fields` или список через запятую. `all` используется отдельно. В OCR попадают только выбранные вырезки; поиск страницы и разметки выполняется целиком. В ответе остаются выбранные поля и `meta`. Неизвестные названия, пустой список и сочетание `all` с другими полями возвращают `422 VALIDATION_ERROR`.

Лимиты текущего сервера — до **75 МиБ** на файл, до **12 000 пикселей** по каждой стороне и до **50 млн пикселей** суммарно.

## Прямой запрос

```sh
export OCR_BASE_URL=http://192.168.3.152:8100
curl --fail-with-body -sS "$OCR_BASE_URL/v1/ocr/egrn" \
  -F 'image=@scan.png' \
  -F 'ocr_model=paddle-cyrillic'
```

Пример ответа HTTP 200 с **синтетическими** значениями. Время в примере иллюстративное.

```json
{
  "success": true,
  "data": {
    "application_number": "0000000042",
    "object_type": "Помещение",
    "cadastral_number": "00:00:0000000:999",
    "cadastral_quarter_number": "00:00:0000000",
    "cadastral_number_assignment_date": "01.01.2020",
    "previous_registration_number": "Данные отсутствуют",
    "address": "Тестовый город, дом 123",
    "floor": "2 Этаж",
    "area": "40.5",
    "purpose": "Жилое",
    "object_name": "Квартира",
    "residential_premises_type": "Квартира",
    "cadastral_value": "1000000.00",
    "parent_cadastral_numbers": "00:00:0000000:998",
    "cadastral_engineer": "Тестов Иван Иванович",
    "cultural_heritage": "Данные отсутствуют",
    "claims": "Данные отсутствуют",
    "specialized_housing": "Данные отсутствуют",
    "actuality_date": "05.11.2026",
    "meta": {
      "engine": "paddleocr",
      "model": "paddle-cyrillic",
      "duration_seconds": 1.8
    }
  },
  "error": null
}
```

| Ключ в `data` | Что получает клиент |
| --- | --- |
| `application_number` | Номер заявления из шапки. Строка цифр сохраняет ведущие нули. |
| `object_type` | Вид объекта из первой строки верхней таблицы. |
| `cadastral_number` | Кадастровый номер объекта из верхней таблицы. |
| `cadastral_quarter_number` | Номер кадастрового квартала. |
| `cadastral_number_assignment_date` | Дата присвоения кадастрового номера, как прочитана в таблице. |
| `previous_registration_number` | Ранее присвоенный государственный учётный номер. |
| `address` | Значение «Адрес (местоположение)». Строки адреса соединяются пробелом. |
| `floor` | Номер и тип этажа. |
| `area` | Площадь в квадратных метрах, строка без преобразования в число. |
| `purpose` | Назначение объекта. |
| `object_name` | Значение строки «Наименование». |
| `residential_premises_type` | Вид жилого помещения. |
| `cadastral_value` | Кадастровая стоимость в рублях, строка без преобразования в число. |
| `parent_cadastral_numbers` | Кадастровые номера иных объектов, в пределах которых расположен объект. |
| `cadastral_engineer` | Сведения о кадастровом инженере. |
| `cultural_heritage` | Сведения о включении объекта в реестр объектов культурного наследия. |
| `claims` | Сведения о правопритязаниях. |
| `specialized_housing` | Сведения об отнесении помещения к специализированному жилищному фонду и другим видам, указанным в последней строке таблицы. |
| `actuality_date` | Дата из подписи «Данные актуальны на» в формате `ДД.ММ.ГГГГ`. |
| `meta` | Движок, выбранная модель и длительность обработки в секундах без ожидания в очереди. |

Все значения полей — строки или `null`. Текст «Данные отсутствуют» сохраняется строкой; переносы строк и повторные пробелы объединяются в один пробел. Незапрошенные поля отсутствуют. При `all` (или явном выборе всех 19 полей), если ни одного пригодного значения нет, API возвращает ошибку. При выборе части полей успешный ответ может содержать только `null`: это означает, что нужные поля не найдены или не прочитаны. Ошибки чтения изображения, неподдерживаемый макет и сбой OCR остаются ошибками. Перед использованием проверьте поля, обязательные для вашего приложения. OCR может вернуть неполный адрес или ошибиться в символе даже при HTTP 200.

Например, только кадастровый номер, площадь и стоимость:

```sh
curl --fail-with-body -sS "$OCR_BASE_URL/v1/ocr/egrn" \
  -F 'image=@scan.png' \
  -F 'fields=cadastral_number' -F 'fields=area' -F 'fields=cadastral_value'
```

Те же `fields` можно передать при создании задачи; готовый результат сохранит выбранный состав полей.

## Обработка через очередь

Создайте задачу одним запросом. Сохраните идентификатор из ответа HTTP 202.

```sh
curl --fail-with-body -sS "$OCR_BASE_URL/v1/tasks/egrn" -F 'image=@scan.png'
```

```json
{"success":true,"data":{"task_id":"00000000-0000-4000-8000-000000000042","status":"queued"},"error":null}
```

Подставьте фактический `task_id`. В статусе поле `data.operation` равно `egrn`.

```sh
TASK_ID='идентификатор-из-ответа'
curl -sS "$OCR_BASE_URL/v1/tasks/$TASK_ID"
# Когда data.status равен done, заберите готовый JSON.
curl --fail-with-body -sS "$OCR_BASE_URL/v1/tasks/$TASK_ID/result"
# Чтобы отменить задачу, пока она не завершилась:
curl -X POST -sS "$OCR_BASE_URL/v1/tasks/$TASK_ID/cancel"
```

| `data.status` | Действие клиента |
| --- | --- |
| `queued`, `processing` | Опросить статус снова через 1–2 секунды. |
| `done` | Вызвать `GET /v1/tasks/{task_id}/result`. |
| `failed`, `timeout` | Прочитать `data.error.code` и `data.error.message` в ответе статуса. |
| `cancelled` | Завершить ожидание. |

Создание задачи возвращает новый идентификатор при каждом успешном POST. Для повторной проверки существующей задачи используйте GET с сохранённым идентификатором.

### Рабочий пример на Python

Требуется пакет `requests`. Функция возвращает объект `data` одного формата в обоих режимах. `wait_seconds` задаёт срок ожидания клиентом и не отменяет задачу на сервере.

```python
import os
import time
from pathlib import Path

import requests

BASE = os.getenv("OCR_BASE_URL", "http://192.168.3.152:8100").rstrip("/")


def unpack(response):
    body = response.json()
    if not response.ok or not body["success"]:
        error = body.get("error") or {}
        raise RuntimeError(f"HTTP {response.status_code}: {error}")
    return body["data"]


def recognize_egrn(
    path, *, queued=False, model="paddle-cyrillic", fields=None, wait_seconds=120
):
    path = Path(path)
    route = "/v1/tasks/egrn" if queued else "/v1/ocr/egrn"
    with requests.Session() as client:
        with path.open("rb") as stream:
            data = unpack(client.post(
                BASE + route,
                files={"image": (path.name, stream)},
                data={"ocr_model": model, "fields": ["all"] if fields is None else fields},
                timeout=(5, 90),
            ))
        if not queued:
            return data

        task_id = data["task_id"]  # В приложении сохраните до начала опроса.
        url = f"{BASE}/v1/tasks/{task_id}"
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            state = unpack(client.get(url, timeout=(5, 15)))
            if state["status"] == "done":
                return unpack(client.get(url + "/result", timeout=(5, 30)))
            if state["status"] in {"failed", "timeout", "cancelled"}:
                raise RuntimeError(f"{task_id}: {state['status']}, {state['error']}")
            time.sleep(1)
        raise TimeoutError(f"Продолжите опрос задачи {task_id}")


# result = recognize_egrn("scan.png", queued=True)
# print(result["cadastral_number"], result["address"])
```

## Ошибки

Пример при неподдерживаемом макете (HTTP 422, текст сообщения может отличаться).

```json
{"success":false,"data":null,"error":{"code":"OCR_FAILED","message":"Не удалось прочитать поля ЕГРН"}}
```

| HTTP | Код | Действие |
| --- | --- | --- |
| 400 | `UNSUPPORTED_FORMAT`, `IMAGE_READ_ERROR`, `EMPTY_IMAGE` | Проверить расширение, формат и содержимое файла. |
| 413 | `IMAGE_TOO_LARGE` | Уменьшить файл или разрешение. |
| 422 | `VALIDATION_ERROR` | Проверить `image`, `ocr_model`, список `fields` и формат UUID. |
| 422 | `OCR_FAILED` | Проверить страницу, макет и качество скана. |
| 404 | `TASK_NOT_FOUND` | Проверить идентификатор и адрес сервера. |
| 409 | `TASK_NOT_READY`, `TASK_CANCELLED` | Проверить статус перед запросом результата. |
| 429 | `QUEUE_FULL` | Повторить создание задачи позже. |
| 503, 504 | `OCR_ENGINE_UNAVAILABLE`, `OCR_TIMEOUT` | Проверить доступность движка или повторить запрос позже. |

Для диагностики доступны `POST /v1/ocr/crop/egrn` (выровненная страница PNG) и `POST /v1/ocr/crop/egrn-fields` (вырезки, координаты и результаты по участкам). Для получения всех 19 текстовых полей или выбранной части клиенту достаточно одного вызова `/v1/ocr/egrn` либо одного задания `/v1/tasks/egrn`.
