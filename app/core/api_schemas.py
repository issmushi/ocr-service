from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def error_example(code: str, message: str) -> dict[str, Any]:
    return {
        "summary": code,
        "value": {
            "success": False,
            "data": None,
            "error": {"code": code, "message": message},
        },
    }


def error_response(
    description: str,
    examples: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model": ApiResponseSchema,
        "description": description,
        "content": {"application/json": {"examples": examples}},
    }


def json_response(
    description: str,
    examples: dict[str, dict[str, Any]],
    model: type[BaseModel] | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "description": description,
        "content": {"application/json": {"examples": examples}},
    }
    if model is not None:
        response["model"] = model
    return response


def success_example(summary: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": summary,
        "value": {"success": True, "data": data, "error": None},
    }


class ErrorSchema(BaseModel):
    code: str = Field(..., description="Машиночитаемый код ошибки.")
    message: str = Field(..., description="Человекочитаемое описание ошибки.")


class ApiResponseSchema(BaseModel):
    success: bool = Field(..., description="Флаг успешности операции.")
    data: dict[str, Any] | None = Field(
        None, description="Полезная нагрузка ответа."
    )
    error: ErrorSchema | None = Field(
        None,
        description="Ошибка. Если операция успешна, значение равно null.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {"text": "Распознанный текст"},
                "error": None,
            }
        }
    )


class HealthComponentSchema(BaseModel):
    status: str = Field(..., description="Состояние компонента: ok или error.")
    latency_ms: int = Field(..., description="Время проверки в миллисекундах.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Дополнительные безопасные данные компонента.",
    )
    error: str | None = Field(
        None, description="Причина недоступности или null."
    )


class HealthComponentsSchema(BaseModel):
    api: HealthComponentSchema
    postgres: HealthComponentSchema
    rabbitmq: HealthComponentSchema
    worker: HealthComponentSchema
    paddle_ocr: HealthComponentSchema
    russian_ocr: HealthComponentSchema
    mrz_scanner: HealthComponentSchema
    qwen_vlm_service: HealthComponentSchema
    qr: HealthComponentSchema


class HealthDataSchema(BaseModel):
    status: str = Field(
        ...,
        description="Общее состояние OCR-стека: ok или degraded.",
    )
    service: str = Field(..., description="Название сервиса.")
    version: str = Field(..., description="Версия API из pyproject.toml.")
    queue: str = Field(..., description="Состояние подключения к RabbitMQ.")
    workers: int = Field(..., description="Настроенное количество workers.")
    active_workers: int = Field(
        ...,
        description="Число workers, подключённых к очереди RabbitMQ.",
    )
    components: HealthComponentsSchema


def _healthy_components_example() -> dict[str, dict[str, Any]]:
    return {
        "api": {
            "status": "ok",
            "latency_ms": 0,
            "details": {"version": "0.1.0"},
            "error": None,
        },
        "postgres": {
            "status": "ok",
            "latency_ms": 4,
            "details": {},
            "error": None,
        },
        "rabbitmq": {
            "status": "ok",
            "latency_ms": 5,
            "details": {
                "queue_name": "ocr.jobs",
                "message_count": 0,
                "consumer_count": 1,
                "dead_letter_queue_name": "ocr.jobs.dlq",
                "dead_letter_message_count": 0,
            },
            "error": None,
        },
        "worker": {
            "status": "ok",
            "latency_ms": 0,
            "details": {"configured_workers": 1, "active_workers": 1},
            "error": None,
        },
        "paddle_ocr": {
            "status": "ok",
            "latency_ms": 0,
            "details": {
                "engine": "paddleocr",
                "status": "ready",
                "busy": False,
                "ready": True,
                "worker_pid": 12345,
                "device": "cpu",
            },
            "error": None,
        },
        "russian_ocr": {
            "status": "ok",
            "latency_ms": 0,
            "details": {
                "engine": "russian-docs-ocr",
                "status": "ready",
                "busy": False,
                "ready": True,
                "worker_pid": 12346,
                "device": "cpu",
                "model_format": "ONNX",
            },
            "error": None,
        },
        "mrz_scanner": {
            "status": "ok",
            "latency_ms": 0,
            "details": {
                "engine": "mrz-scanner",
                "status": "idle",
                "busy": False,
                "ready": False,
                "worker_pid": None,
            },
            "error": None,
        },
        "qwen_vlm_service": {
            "status": "ok",
            "latency_ms": 2,
            "details": {
                "engine": "qwen3-vl",
                "status": "ready",
                "models": ["qwen3-vl-4b", "qwen3-vl-8b"],
                "active_model": None,
            },
            "error": None,
        },
        "qr": {
            "status": "ok",
            "latency_ms": 3,
            "details": {
                "engine": "opencv-qrcode",
                "encryption": "A256GCM",
                "key_id": "primary",
            },
            "error": None,
        },
    }


class HealthResponseSchema(BaseModel):
    success: bool = Field(..., description="Флаг успешности health-check.")
    data: HealthDataSchema
    error: ErrorSchema | None = Field(None, description="Ошибка или null.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {
                    "status": "ok",
                    "service": "ocr-service",
                    "version": "0.1.0",
                    "queue": "ok",
                    "workers": 1,
                    "active_workers": 1,
                    "components": _healthy_components_example(),
                },
                "error": None,
            }
        }
    )


HEALTH_OK_RESPONSE = json_response(
    "Все компоненты OCR-стека доступны.",
    {
        "healthy": {
            "summary": "Все компоненты работают",
            "value": HealthResponseSchema.model_config["json_schema_extra"][
                "example"
            ],
        }
    },
    model=HealthResponseSchema,
)

_degraded_components = _healthy_components_example()
_degraded_components["paddle_ocr"] = {
    "status": "error",
    "latency_ms": 3001,
    "details": {
        "engine": "paddleocr",
        "status": "down",
        "busy": False,
        "ready": False,
        "worker_pid": None,
        "device": "cpu",
    },
    "error": "PaddleOCR недоступен",
}

HEALTH_UNAVAILABLE_RESPONSE = json_response(
    "Один или несколько компонентов OCR-стека недоступны.",
    {
        "degraded": {
            "summary": "Стек работает не полностью",
            "value": {
                "success": False,
                "data": {
                    "status": "degraded",
                    "service": "ocr-service",
                    "version": "0.1.0",
                    "queue": "ok",
                    "workers": 1,
                    "active_workers": 1,
                    "components": _degraded_components,
                },
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "Недоступны компоненты: paddle_ocr",
                },
            },
        }
    },
    model=HealthResponseSchema,
)


COMMON_ERROR_RESPONSES = {
    400: error_response(
        "Некорректный запрос.",
        {
            "no_image": error_example("NO_IMAGE", "В запросе нет поля image"),
            "empty_image": error_example(
                "EMPTY_IMAGE", "Передано пустое изображение"
            ),
            "unsupported_format": error_example(
                "UNSUPPORTED_FORMAT", "Неподдерживаемый формат изображения"
            ),
            "image_read_error": error_example(
                "IMAGE_READ_ERROR", "Изображение невозможно прочитать"
            ),
            "empty_payload": error_example(
                "EMPTY_PAYLOAD", "Некорректный запрос: проверьте поля"
            ),
        },
    ),
    413: error_response(
        "Изображение превышает допустимый размер.",
        {
            "image_too_large": error_example(
                "IMAGE_TOO_LARGE", "Изображение превышает лимит 75 МБ"
            )
        },
    ),
    429: error_response(
        "Очередь OCR-задач переполнена.",
        {
            "queue_full": error_example(
                "QUEUE_FULL", "Очередь переполнена: 50/50 активных задач"
            )
        },
    ),
    500: error_response(
        "Непредвиденная ошибка сервера.",
        {
            "internal_error": error_example(
                "INTERNAL_ERROR", "Непредвиденная ошибка"
            )
        },
    ),
}

QR_DECODE_ERROR_RESPONSES = {
    400: error_response(
        "Некорректный запрос с изображением QR.",
        {
            "no_image": error_example("NO_IMAGE", "В запросе нет поля image"),
            "empty_image": error_example(
                "EMPTY_IMAGE", "Передано пустое изображение"
            ),
            "image_read_error": error_example(
                "IMAGE_READ_ERROR", "Изображение невозможно прочитать"
            ),
        },
    ),
    413: COMMON_ERROR_RESPONSES[413],
    500: COMMON_ERROR_RESPONSES[500],
}

PAYLOAD_ERROR_RESPONSE = error_response(
    "Некорректный JSON-запрос.",
    {
        "empty_payload": error_example(
            "EMPTY_PAYLOAD", "Некорректный запрос: проверьте поля"
        )
    },
)
OCR_FAILED_RESPONSE = error_response(
    "OCR не смог получить пригодный результат.",
    {
        "ocr_failed": error_example(
            "OCR_FAILED", "OCR не смог получить пригодный результат"
        )
    },
)
OCR_TIMEOUT_RESPONSE = error_response(
    "Превышен таймаут обработки.",
    {
        "ocr_timeout": error_example(
            "OCR_TIMEOUT", "Превышен таймаут обработки"
        )
    },
)
OCR_ENGINE_UNAVAILABLE_RESPONSE = error_response(
    "OCR engine недоступен.",
    {
        "ocr_engine_unavailable": error_example(
            "OCR_ENGINE_UNAVAILABLE",
            "OCR engine не инициализирован или недоступен",
        )
    },
)
QR_DECODE_FAILED_RESPONSE = error_response(
    (
        "QR-код не найден, содержит некорректные данные "
        "или передано слишком много кадров."
    ),
    {
        "qr_not_found": error_example("QR_NOT_FOUND", "QR-код не найден"),
        "qr_invalid_payload": error_example(
            "QR_INVALID_PAYLOAD", "QR содержит некорректные данные"
        ),
        "too_many_images": error_example(
            "VALIDATION_ERROR", "Можно передать не более 3 изображений"
        ),
    },
)
QR_SERVICE_UNAVAILABLE_RESPONSE = error_response(
    "QR-модуль недоступен.",
    {
        "qr_service_unavailable": error_example(
            "QR_SERVICE_UNAVAILABLE", "QR-модуль недоступен"
        )
    },
)
VALIDATION_ERROR_RESPONSE = error_response(
    "Некорректные параметры запроса.",
    {
        "validation_error": error_example(
            "VALIDATION_ERROR", "Некорректные параметры запроса"
        )
    },
)

QR_DECODE_RESPONSE = json_response(
    "QR-код распознан.",
    {
        "decoded": success_example(
            "QR распознан",
            {
                "passport": {
                    "surname": "ТЕСТОВ",
                    "name": "ТЕСТ",
                    "patronymic": "ТЕСТОВИЧ",
                    "sex": "МУЖ.",
                    "date_of_birth": "15.01.1990",
                    "passport_series": "45 15",
                    "passport_number": "123456",
                },
                "meta": {
                    "format": "passport",
                    "version": 2,
                    "encrypted": True,
                    "algorithm": "A256GCM",
                    "key_id": "primary",
                    "qr_count": 1,
                },
                "filename": "qr.png",
                "content_type": "image/png",
            },
        )
    },
    model=ApiResponseSchema,
)
