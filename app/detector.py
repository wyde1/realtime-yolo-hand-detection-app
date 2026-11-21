"""YOLO11s hand finger-number detector and image encoding/decoding utilities."""

from __future__ import annotations

import base64
import binascii
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Single detected hand gesture bounding box."""

    label: str
    class_id: int
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": [round(coord, 2) for coord in self.bbox],
        }


@dataclass(frozen=True)
class InferenceResult:
    """Output of running detection on a single BGR frame."""

    annotated_bgr: np.ndarray
    detections: list[Detection]
    inference_ms: float


@runtime_checkable
class HandDetector(Protocol):
    """Protocol implemented by both YOLOHandDetector and test mocks."""

    @property
    def class_names(self) -> dict[int, str]: ...

    def predict(self, frame_bgr: np.ndarray, conf: float | None = None) -> InferenceResult: ...


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes (JPEG/PNG/WebP) into an OpenCV BGR array."""
    if not data:
        raise ValueError("Empty image payload.")
    np_arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Unable to decode image bytes. Ensure a valid JPEG or PNG image is sent.")
    return image


def decode_data_uri(payload: str) -> np.ndarray:
    """Decode a base64 data URI (or raw base64 string) into a BGR numpy array."""
    if not payload or not isinstance(payload, str):
        raise ValueError("Empty base64 payload.")

    encoded = payload.split(",", 1)[1] if "," in payload else payload
    if not encoded.strip():
        raise ValueError("Base64 image payload is empty after header.")

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image payload.") from exc

    return decode_image_bytes(image_bytes)


def encode_image_bytes(
    image_bgr: np.ndarray,
    ext: str = ".jpg",
    jpeg_quality: int = 85,
) -> bytes:
    """Encode a BGR numpy array to JPEG or PNG bytes."""
    params: list[int] = []
    if ext.lower() in {".jpg", ".jpeg"}:
        quality = max(10, min(100, int(jpeg_quality)))
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(ext, image_bgr, params)
    if not success:
        raise RuntimeError(f"Failed to encode image as {ext}.")
    return buffer.tobytes()


def encode_data_uri(image_bgr: np.ndarray, jpeg_quality: int = 85) -> str:
    """Encode a BGR numpy array as a JPEG base64 data URI."""
    encoded_bytes = encode_image_bytes(image_bgr, ext=".jpg", jpeg_quality=jpeg_quality)
    b64 = base64.b64encode(encoded_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class YOLOHandDetector:
    """Wrapper around Ultralytics YOLO11s for hand finger-number (1-5) detection."""

    def __init__(self, model_path: Path, default_conf: float = 0.45) -> None:
        self.model_path = Path(model_path)
        self.default_conf = default_conf
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found at '{self.model_path}'. "
                "Run 'python scripts/download_weights.py' or place 'best.pt' in the weights/ directory."
            )

        # Lazy import keeps app imports lightweight when using mocked detectors in tests.
        from ultralytics import YOLO

        self._model = YOLO(str(self.model_path))
        self._predict_lock = threading.Lock()

    @property
    def class_names(self) -> dict[int, str]:
        names = getattr(self._model, "names", {})
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {idx: str(name) for idx, name in enumerate(names)}

    def predict(self, frame_bgr: np.ndarray, conf: float | None = None) -> InferenceResult:
        """Run inference on a single BGR frame and return annotated image + detections."""
        threshold = self.default_conf if conf is None else max(0.05, min(0.95, float(conf)))
        with self._predict_lock:
            start = time.perf_counter()
            results = self._model(frame_bgr, conf=threshold, verbose=False)
            result = results[0]
            annotated = result.plot()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        names = self.class_names

        detections: list[Detection] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy_list = result.boxes.xyxy.cpu().tolist()
            conf_list = result.boxes.conf.cpu().tolist()
            cls_list = result.boxes.cls.cpu().tolist()

            for coords, score, cls_raw in zip(xyxy_list, conf_list, cls_list):
                cls_id = int(cls_raw)
                detections.append(
                    Detection(
                        label=names.get(cls_id, str(cls_id)),
                        class_id=cls_id,
                        confidence=float(score),
                        bbox=[float(c) for c in coords],
                    )
                )

        # Sort highest confidence first for deterministic API responses.
        detections.sort(key=lambda det: det.confidence, reverse=True)
        return InferenceResult(
            annotated_bgr=annotated,
            detections=detections,
            inference_ms=round(elapsed_ms, 2),
        )
