import pytest
from fastapi.testclient import TestClient

from app.modules.vlm.server import VlmBackendRuntime
from app.modules.vlm.server import create_app as vlm_app
from tests.test_vlm_backend import FakeEngine as QwenFake
from tests.test_vlm_backend import image_bytes


@pytest.mark.parametrize("model", ["qwen3-vl-4b", "qwen3-vl-8b"])
@pytest.mark.parametrize("device", ["cpu", "unknown"])
def test_qwen_cpu_request_checks_selected_engine_before_recognition(
    model, device
):
    events = []
    engine = QwenFake(model, events)
    engine.runtime_info = {"device": device}
    with TestClient(vlm_app(VlmBackendRuntime(lambda _: engine))) as client:
        response = client.post(
            "/recognize",
            files={"image": ("mrz.png", image_bytes())},
            data={"model": model, "cpu_only": "true"},
        )
        assert response.status_code == (200 if device == "cpu" else 503)
        if device == "cpu":
            assert response.json()["device"] == "cpu"
        else:
            assert all(operation != "recognize" for _, operation, _ in events)
