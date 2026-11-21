"""FastAPI server for real-time YOLO11s hand finger-number (1-5) detection."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import PROJECT_ROOT, STATIC_DIR, Settings
from app.detector import (
    HandDetector,
    YOLOHandDetector,
    decode_data_uri,
    decode_image_bytes,
    encode_data_uri,
    encode_image_bytes,
)

logger = logging.getLogger(__name__)


def _parse_conf(raw: Any, default: float) -> float:
    if raw is None or isinstance(raw, bool):
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(val):
        return default
    return max(0.05, min(0.95, val))


def _parse_quality(raw: Any, default: int) -> int:
    if raw is None or isinstance(raw, bool):
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(val):
        return default
    return max(10, min(95, int(round(val))))


def _safe_model_display_path(model_path: Path) -> str:
    """Return a workspace-relative path or filename to avoid exposing server directories."""
    try:
        return model_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return model_path.name


def _require_detector(app: FastAPI) -> HandDetector:
    detector: HandDetector | None = getattr(app.state, "detector", None)
    if detector is None:
        load_error = getattr(app.state, "load_error", "Model weights are not loaded.")
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {load_error}",
        )
    return detector


def create_app(
    detector: HandDetector | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app_settings = settings or Settings.from_env()
    display_model_path = _safe_model_display_path(app_settings.model_path)

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        if detector is not None:
            fastapi_app.state.detector = detector
            fastapi_app.state.load_error = None
        else:
            try:
                fastapi_app.state.detector = YOLOHandDetector(
                    model_path=app_settings.model_path,
                    default_conf=app_settings.conf_threshold,
                )
                fastapi_app.state.load_error = None
                logger.info("Loaded YOLO model from %s", display_model_path)
            except FileNotFoundError:
                fastapi_app.state.detector = None
                fastapi_app.state.load_error = (
                    f"Model weights not found ({display_model_path}). "
                    "Run 'python scripts/download_weights.py' or place 'best.pt' in weights/."
                )
                logger.warning("Model weights not found at %s", app_settings.model_path)
            except Exception:  # pragma: no cover - unexpected model load failure
                fastapi_app.state.detector = None
                fastapi_app.state.load_error = "Failed to initialize YOLO model weights. Check server logs."
                logger.exception("Unexpected error while loading YOLO model from %s", app_settings.model_path)
        yield

    fastapi_app = FastAPI(
        title="Real-Time YOLO11s Hand Finger-Number Detection API",
        description=(
            "FastAPI REST and WebSocket backend serving a fine-tuned YOLO11s model "
            "for hand finger-number (1-5) detection."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    fastapi_app.state.settings = app_settings
    fastapi_app.state.detector = detector
    fastapi_app.state.load_error = None

    fastapi_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @fastapi_app.get("/", include_in_schema=False)
    async def get_index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @fastapi_app.get("/favicon.ico", include_in_schema=False)
    async def get_favicon() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.ico")

    @fastapi_app.get("/health")
    async def health_check() -> JSONResponse:
        active_detector: HandDetector | None = getattr(fastapi_app.state, "detector", None)
        if active_detector is None:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "degraded",
                    "model_loaded": False,
                    "model_path": display_model_path,
                    "classes": [],
                    "default_conf": app_settings.conf_threshold,
                    "detail": getattr(fastapi_app.state, "load_error", None),
                },
            )
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "model_loaded": True,
                "model_path": display_model_path,
                "classes": list(active_detector.class_names.values()),
                "default_conf": app_settings.conf_threshold,
            },
        )

    async def _read_and_decode_upload(file: UploadFile) -> Any:
        # Read at most max_image_bytes + 1 so oversized uploads are rejected without buffering gigabytes.
        content = await file.read(app_settings.max_image_bytes + 1)
        if len(content) > app_settings.max_image_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds maximum allowed size of {app_settings.max_image_bytes} bytes.",
            )
        try:
            return await asyncio.to_thread(decode_image_bytes, content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @fastapi_app.post(
        "/detect",
        response_class=Response,
        responses={200: {"content": {"image/png": {}}}},
    )
    async def detect_image(
        file: UploadFile = File(...),
        conf: float | None = Query(default=None, ge=0.05, le=0.95),
    ) -> Response:
        """Run YOLO11s detection on an uploaded image and return the annotated PNG."""
        active_detector = _require_detector(fastapi_app)
        image_bgr = await _read_and_decode_upload(file)
        threshold = _parse_conf(conf, app_settings.conf_threshold)

        result = await asyncio.to_thread(active_detector.predict, image_bgr, threshold)
        png_bytes = await asyncio.to_thread(encode_image_bytes, result.annotated_bgr, ".png")

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "X-Detections-Count": str(len(result.detections)),
                "X-Inference-Ms": f"{result.inference_ms:.2f}",
            },
        )

    @fastapi_app.post("/detect/json")
    async def detect_json(
        file: UploadFile = File(...),
        conf: float | None = Query(default=None, ge=0.05, le=0.95),
    ) -> JSONResponse:
        """Run YOLO11s detection on an uploaded image and return structured JSON predictions."""
        active_detector = _require_detector(fastapi_app)
        image_bgr = await _read_and_decode_upload(file)
        threshold = _parse_conf(conf, app_settings.conf_threshold)

        result = await asyncio.to_thread(active_detector.predict, image_bgr, threshold)
        height, width = image_bgr.shape[:2]
        return JSONResponse(
            content={
                "detections": [det.to_dict() for det in result.detections],
                "count": len(result.detections),
                "inference_ms": result.inference_ms,
                "image_size": {"width": int(width), "height": int(height)},
            }
        )

    @fastapi_app.websocket("/ws")
    async def websocket_json_endpoint(websocket: WebSocket) -> None:
        """Real-time base64 WebSocket endpoint used by the browser UI.

        Accepts either:
        - JSON text: ``{"image": "data:image/jpeg;base64,...", "conf": 0.45, "quality": 80}``
        - Plain data URI text: ``data:image/jpeg;base64,...``

        Responds with JSON containing the annotated JPEG data URI, detections, and timing.
        """
        await websocket.accept()
        active_detector: HandDetector | None = getattr(fastapi_app.state, "detector", None)
        if active_detector is None:
            await websocket.send_json(
                {
                    "error": getattr(
                        fastapi_app.state,
                        "load_error",
                        "Model weights are not loaded on the server.",
                    )
                }
            )
            await websocket.close(code=1011)
            return

        try:
            while True:
                try:
                    raw_text = await websocket.receive_text()
                except KeyError:
                    await websocket.send_json({"error": "Expected text frame on /ws."})
                    continue

                if len(raw_text.encode("utf-8")) > app_settings.max_image_bytes * 2:
                    await websocket.send_json({"error": "Frame payload exceeds size limit."})
                    continue

                conf = app_settings.conf_threshold
                quality = app_settings.jpeg_quality
                image_payload = raw_text

                if raw_text.lstrip().startswith("{"):
                    try:
                        message = json.loads(raw_text)
                    except json.JSONDecodeError:
                        await websocket.send_json({"error": "Malformed JSON message."})
                        continue
                    if not isinstance(message, dict):
                        await websocket.send_json({"error": "JSON payload must be an object."})
                        continue
                    image_payload = message.get("image", "")
                    conf = _parse_conf(message.get("conf"), app_settings.conf_threshold)
                    quality = _parse_quality(message.get("quality"), quality)

                try:
                    frame_bgr = await asyncio.to_thread(decode_data_uri, image_payload)
                except ValueError as exc:
                    await websocket.send_json({"error": str(exc)})
                    continue

                try:
                    result = await asyncio.to_thread(active_detector.predict, frame_bgr, conf)
                    annotated_uri = await asyncio.to_thread(
                        encode_data_uri,
                        result.annotated_bgr,
                        quality,
                    )
                except Exception:
                    logger.exception("Inference failed during /ws frame processing")
                    await websocket.send_json({"error": "Internal inference error."})
                    continue

                try:
                    await websocket.send_json(
                        {
                            "image": annotated_uri,
                            "detections": [det.to_dict() for det in result.detections],
                            "count": len(result.detections),
                            "inference_ms": result.inference_ms,
                        }
                    )
                except RuntimeError:
                    # Client closed the socket while inference was in flight.
                    logger.debug("WebSocket client closed /ws before response could be sent")
                    break
        except WebSocketDisconnect:
            logger.debug("WebSocket client disconnected from /ws")

    @fastapi_app.websocket("/ws/binary")
    async def websocket_binary_endpoint(
        websocket: WebSocket,
        conf: float | None = Query(default=None, ge=0.05, le=0.95),
    ) -> None:
        """Real-time binary WebSocket endpoint for Streamlit and Python clients.

        Receives raw JPEG/PNG bytes and sends back annotated JPEG bytes.
        """
        await websocket.accept()
        active_detector: HandDetector | None = getattr(fastapi_app.state, "detector", None)
        if active_detector is None:
            await websocket.close(code=1011, reason="Model weights are not loaded.")
            return

        threshold = _parse_conf(conf, app_settings.conf_threshold)
        try:
            while True:
                try:
                    raw_bytes = await websocket.receive_bytes()
                except KeyError:
                    continue

                if len(raw_bytes) > app_settings.max_image_bytes:
                    continue
                try:
                    frame_bgr = await asyncio.to_thread(decode_image_bytes, raw_bytes)
                except ValueError:
                    continue

                try:
                    result = await asyncio.to_thread(active_detector.predict, frame_bgr, threshold)
                    jpg_bytes = await asyncio.to_thread(
                        encode_image_bytes,
                        result.annotated_bgr,
                        ".jpg",
                        app_settings.jpeg_quality,
                    )
                except Exception:
                    logger.exception("Inference failed during /ws/binary frame processing")
                    await websocket.close(code=1011, reason="Internal inference error.")
                    break

                try:
                    await websocket.send_bytes(jpg_bytes)
                except RuntimeError:
                    logger.debug("WebSocket client closed /ws/binary before response could be sent")
                    break
        except WebSocketDisconnect:
            logger.debug("WebSocket client disconnected from /ws/binary")

    return fastapi_app


app = create_app()


if __name__ == "__main__":
    cfg = Settings.from_env()
    uvicorn.run("app.main:app", host=cfg.host, port=cfg.port, reload=False)
