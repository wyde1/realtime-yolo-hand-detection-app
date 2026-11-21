"""Application configuration loaded from environment variables and optional .env file."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"


def load_dotenv_file(env_path: Path | None = None) -> None:
    """Load key=value pairs from a local .env file without overriding existing env vars."""
    path = env_path or (PROJECT_ROOT / ".env")
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _parse_float(value: str | None, default: float, min_val: float, max_val: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    if not math.isfinite(parsed):
        return default
    return max(min_val, min(max_val, parsed))


def _parse_int(
    value: str | None,
    default: int,
    min_val: int = 1,
    max_val: int | None = None,
) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    parsed = max(min_val, parsed)
    if max_val is not None:
        parsed = min(max_val, parsed)
    return parsed


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the detection API server."""

    model_path: Path
    conf_threshold: float = 0.45
    max_image_bytes: int = 10 * 1024 * 1024  # 10 MB
    jpeg_quality: int = 85
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        load_dotenv_file(env_file)
        raw_model_path = (os.getenv("MODEL_PATH") or "").strip() or "weights/best.pt"
        model_path = Path(raw_model_path)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        return cls(
            model_path=model_path,
            conf_threshold=_parse_float(os.getenv("CONF_THRESHOLD"), 0.45, 0.05, 0.95),
            max_image_bytes=_parse_int(os.getenv("MAX_IMAGE_BYTES"), 10 * 1024 * 1024, 1024),
            jpeg_quality=_parse_int(os.getenv("JPEG_QUALITY"), 85, 10, 100),
            host=(os.getenv("HOST") or "").strip() or "0.0.0.0",
            port=_parse_int(os.getenv("PORT"), 8000, 1, 65535),
        )
