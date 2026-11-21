"""Unit and integration tests for the FastAPI detection server and weight downloader.

The default test suite uses a mocked detector and passes even when ``weights/best.pt``
is not present locally. Tests marked with ``@pytest.mark.real_model`` exercise the
actual YOLO11s checkpoint when available.
"""

from __future__ import annotations

import base64
import urllib.error
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, Settings
from app.detector import Detection, InferenceResult, YOLOHandDetector
from app.main import create_app
from scripts.download_weights import download_weights


class MockHandDetector:
    """Deterministic fake detector for fast, weight-free API testing."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._names = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
        self._detections = detections if detections is not None else [
            Detection(label="5", class_id=4, confidence=0.94, bbox=[40.0, 30.0, 210.0, 220.0]),
            Detection(label="2", class_id=1, confidence=0.62, bbox=[10.0, 10.0, 90.0, 90.0]),
        ]
        self.last_conf: float | None = None

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._names)

    def predict(self, frame_bgr: np.ndarray, conf: float | None = None) -> InferenceResult:
        self.last_conf = conf
        threshold = 0.45 if conf is None else conf
        filtered = [d for d in self._detections if d.confidence >= threshold]
        annotated = frame_bgr.copy()
        for det in filtered:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return InferenceResult(annotated_bgr=annotated, detections=filtered, inference_ms=4.2)


def _make_jpeg_bytes(width: int = 320, height: int = 240) -> bytes:
    img = np.full((height, width, 3), 120, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _make_data_uri() -> str:
    b64 = base64.b64encode(_make_jpeg_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@pytest.fixture()
def mock_detector() -> MockHandDetector:
    return MockHandDetector()


@pytest.fixture()
def client(mock_detector: MockHandDetector) -> TestClient:
    app = create_app(detector=mock_detector)
    with TestClient(app) as test_client:
        yield test_client


def test_index_and_static_assets(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "YOLO11s Hand Gesture Detector" in resp.text

    for asset_path in (
        "/favicon.ico",
        "/static/styles.css",
        "/static/app.js",
        "/static/site.webmanifest",
        "/static/favicon-16x16.png",
        "/static/favicon-32x32.png",
        "/static/apple-touch-icon.png",
        "/static/android-chrome-192x192.png",
        "/static/android-chrome-512x512.png",
    ):
        asset_resp = client.get(asset_path)
        assert asset_resp.status_code == 200, f"Asset {asset_path} returned {asset_resp.status_code}"


def test_health_loaded(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["classes"] == ["1", "2", "3", "4", "5"]


def test_degraded_server_without_weights(tmp_path: Path) -> None:
    missing_weights = tmp_path / "does_not_exist.pt"
    app = create_app(settings=Settings(model_path=missing_weights))
    with TestClient(app) as degraded_client:
        health = degraded_client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False
        assert str(tmp_path) not in body["model_path"]

        resp = degraded_client.post(
            "/detect",
            files={"file": ("sample.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 503

        json_resp = degraded_client.post(
            "/detect/json",
            files={"file": ("sample.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert json_resp.status_code == 503

        with degraded_client.websocket_connect("/ws") as ws:
            err_msg = ws.receive_json()
            assert "error" in err_msg

        with degraded_client.websocket_connect("/ws/binary") as ws_bin:
            msg = ws_bin.receive()
            assert msg["type"] == "websocket.close"
            assert msg["code"] == 1011


def test_detect_png_endpoint(client: TestClient, mock_detector: MockHandDetector) -> None:
    resp = client.post(
        "/detect?conf=0.70",
        files={"file": ("sample.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.headers["x-detections-count"] == "1"
    assert mock_detector.last_conf == pytest.approx(0.70)

    decoded = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (240, 320)


def test_detect_json_endpoint(client: TestClient) -> None:
    resp = client.post(
        "/detect/json?conf=0.45",
        files={"file": ("sample.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert data["detections"][0]["label"] == "5"
    assert data["image_size"] == {"width": 320, "height": 240}
    assert data["inference_ms"] == pytest.approx(4.2)


def test_detect_invalid_image_returns_400(client: TestClient) -> None:
    resp = client.post(
        "/detect",
        files={"file": ("broken.jpg", b"not-an-image", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_detect_oversized_payload_returns_413(mock_detector: MockHandDetector) -> None:
    tiny_settings = Settings(model_path=Path("unused.pt"), max_image_bytes=128)
    app = create_app(detector=mock_detector, settings=tiny_settings)
    with TestClient(app) as small_client:
        resp = small_client.post(
            "/detect",
            files={"file": ("big.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 413


def test_websocket_json_and_plain_uri(client: TestClient) -> None:
    data_uri = _make_data_uri()
    with client.websocket_connect("/ws") as ws:
        # Structured JSON request
        ws.send_json({"image": data_uri, "conf": 0.80, "quality": 80})
        resp1 = ws.receive_json()
        assert resp1["image"].startswith("data:image/jpeg;base64,")
        assert resp1["count"] == 1
        assert resp1["detections"][0]["label"] == "5"

        # Backward-compatible plain data URI
        ws.send_text(data_uri)
        resp2 = ws.receive_json()
        assert resp2["count"] == 2

        # Malformed payload should return an error JSON without crashing socket
        ws.send_text("data:image/jpeg;base64,!!!invalid!!!")
        err_resp = ws.receive_json()
        assert "error" in err_resp

        # Non-finite numbers (inf / nan) should safely fall back to defaults instead of raising OverflowError
        ws.send_json({"image": data_uri, "conf": "nan", "quality": "inf"})
        safe_resp = ws.receive_json()
        assert safe_resp["image"].startswith("data:image/jpeg;base64,")
        assert safe_resp["count"] == 2


def test_websocket_binary(client: TestClient) -> None:
    jpg_bytes = _make_jpeg_bytes()
    with client.websocket_connect("/ws/binary?conf=0.50") as ws:
        ws.send_bytes(jpg_bytes)
        annotated_bytes = ws.receive_bytes()
        decoded = cv2.imdecode(np.frombuffer(annotated_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape[:2] == (240, 320)


def test_download_weights_404_gives_manual_instructions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _raise_404(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            url="https://example.invalid/best.pt",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", _raise_404)
    target = tmp_path / "weights" / "best.pt"
    with pytest.raises(RuntimeError) as exc_info:
        download_weights("https://example.invalid/best.pt", output_path=target)

    message = str(exc_info.value)
    assert "404" in message
    assert str(target) in message
    assert "MODEL_PATH" in message


# --- Real model integration tests (skipped automatically if weights/best.pt is absent) ---

REAL_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best.pt"


@pytest.mark.real_model
@pytest.mark.skipif(not REAL_WEIGHTS_PATH.is_file(), reason="weights/best.pt not present locally")
def test_real_yolo_model_inference_and_endpoints() -> None:
    detector = YOLOHandDetector(model_path=REAL_WEIGHTS_PATH, default_conf=0.45)
    assert list(detector.class_names.values()) == ["1", "2", "3", "4", "5"]

    app = create_app(detector=detector)
    with TestClient(app) as real_client:
        health = real_client.get("/health").json()
        assert health["status"] == "ok"
        assert health["classes"] == ["1", "2", "3", "4", "5"]

        resp = real_client.post(
            "/detect/json",
            files={"file": ("blank.jpg", _make_jpeg_bytes(640, 480), "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["detections"], list)
        assert body["inference_ms"] > 0

        with real_client.websocket_connect("/ws") as ws:
            ws.send_json({"image": _make_data_uri(), "conf": 0.45, "quality": 70})
            ws_payload = ws.receive_json()
            assert ws_payload["image"].startswith("data:image/jpeg;base64,")
            assert isinstance(ws_payload["detections"], list)
