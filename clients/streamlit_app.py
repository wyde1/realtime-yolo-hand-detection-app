"""Optional Streamlit client streaming webcam frames to the FastAPI /ws/binary endpoint."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import cv2
import numpy as np
import streamlit as st
import websocket


def _load_dotenv() -> None:
    """Load optional .env from the repository root without overriding env vars."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
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


class BinaryWebSocketClient:
    """Thread-safe WebSocket client exchanging raw JPEG frames with backpressure."""

    def __init__(self, url: str, inflight_timeout_s: float = 2.5) -> None:
        self.url = url
        self.inflight_timeout_s = inflight_timeout_s
        self._lock = threading.Lock()
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._connected = False
        self._closed = False
        self._in_flight = False
        self._last_sent_at = 0.0
        self._latest_frame: bytes | None = None
        self._last_error: str | None = None

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def can_send(self) -> bool:
        """Allow at most one frame in flight at a time (with timeout recovery)."""
        with self._lock:
            if not self._connected:
                return False
            if not self._in_flight:
                return True
            if (time.monotonic() - self._last_sent_at) >= self.inflight_timeout_s:
                self._in_flight = False
                return True
            return False

    def pop_latest_frame(self) -> bytes | None:
        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None
            return frame

    def start(self, timeout_s: float = 3.0) -> bool:
        def _on_open(_ws: websocket.WebSocketApp) -> None:
            with self._lock:
                self._connected = True
                self._closed = False
                self._in_flight = False
                self._last_error = None

        def _on_message(_ws: websocket.WebSocketApp, message: bytes | str) -> None:
            if isinstance(message, (bytes, bytearray)):
                with self._lock:
                    self._latest_frame = bytes(message)
                    self._in_flight = False

        def _on_error(_ws: websocket.WebSocketApp, error: Exception) -> None:
            with self._lock:
                self._last_error = str(error)
                self._in_flight = False

        def _on_close(_ws: websocket.WebSocketApp, code: int | None, msg: str | None) -> None:
            with self._lock:
                self._connected = False
                self._closed = True
                self._in_flight = False
                if code and code != 1000:
                    detail = f" ({msg})" if msg else ""
                    self._last_error = f"Server closed WebSocket with code {code}{detail}."

        ws_app = websocket.WebSocketApp(
            self.url,
            on_open=_on_open,
            on_message=_on_message,
            on_error=_on_error,
            on_close=_on_close,
        )
        with self._lock:
            self._ws = ws_app

        self._thread = threading.Thread(target=ws_app.run_forever, daemon=True)
        self._thread.start()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._connected or self._closed or self._last_error:
                    break
            time.sleep(0.05)

        # Wait briefly to catch immediate server rejection (e.g., code 1011 missing weights).
        if self.connected:
            time.sleep(0.1)
        return self.connected

    def send_frame(self, jpg_bytes: bytes) -> bool:
        with self._lock:
            ws_app = self._ws
            if ws_app is None or not self._connected:
                return False
            self._in_flight = True
            self._last_sent_at = time.monotonic()

        try:
            ws_app.send(jpg_bytes, opcode=websocket.ABNF.OPCODE_BINARY)
            return True
        except Exception as exc:
            with self._lock:
                self._in_flight = False
                self._last_error = str(exc)
            raise

    def close(self) -> None:
        with self._lock:
            ws_app = self._ws
            self._ws = None
            self._connected = False
            self._in_flight = False
        if ws_app is not None:
            try:
                ws_app.close()
            except Exception:
                pass


def build_ws_url(base_url: str, conf: float) -> str:
    """Safely set or replace the ``conf`` query parameter on a WebSocket URL."""
    cleaned = base_url.strip()
    parts = urlsplit(cleaned)
    query_items = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "conf"]
    query_items.append(("conf", f"{conf:.2f}"))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query_items), parts.fragment))


def main() -> None:
    _load_dotenv()
    st.set_page_config(page_title="YOLO11s Hand Detection (Streamlit Client)", layout="centered")
    st.title("YOLO11s Hand Gesture Detection — Streamlit Client")
    st.caption("Streams local webcam frames over binary WebSocket (`/ws/binary`) to the FastAPI backend.")

    default_ws = os.getenv("BACKEND_WS_URL", "ws://127.0.0.1:8000/ws/binary")
    with st.sidebar:
        st.header("Connection & Camera")
        backend_ws_url = st.text_input("Backend WebSocket URL", value=default_ws)
        camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
        conf_threshold = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.45, step=0.05)
        jpeg_quality = st.slider("JPEG quality", min_value=30, max_value=95, value=75, step=5)
        fps_cap = st.selectbox("FPS cap", options=[5, 10, 15, 20], index=1)

    run_stream = st.toggle("Run live webcam detection", value=False)
    status_box = st.empty()
    frame_placeholder = st.empty()

    if not run_stream:
        status_box.info("Toggle **Run live webcam detection** to connect to the FastAPI backend.")
        return

    ws_url = build_ws_url(backend_ws_url, conf_threshold)
    client = BinaryWebSocketClient(ws_url)
    if not client.start():
        err_msg = client.last_error or "Connection timed out."
        status_box.error(f"Could not connect to `{ws_url}`: {err_msg}")
        client.close()
        return

    cap = cv2.VideoCapture(int(camera_index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        status_box.error(f"Unable to open webcam index {camera_index}.")
        client.close()
        return

    status_box.success(f"Connected to `{ws_url}`")
    frame_interval = 1.0 / float(fps_cap)

    try:
        while run_stream and client.connected:
            loop_start = time.monotonic()
            ok, frame_bgr = cap.read()
            if not ok:
                status_box.error("Failed to read frame from webcam.")
                break

            if client.can_send():
                encoded_ok, buffer = cv2.imencode(
                    ".jpg",
                    frame_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                if encoded_ok:
                    try:
                        client.send_frame(buffer.tobytes())
                    except Exception as exc:
                        status_box.error(f"WebSocket send failed: {exc}")
                        break

            annotated_bytes = client.pop_latest_frame()
            if annotated_bytes is not None:
                np_arr = np.frombuffer(annotated_bytes, dtype=np.uint8)
                annotated_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if annotated_bgr is not None:
                    # OpenCV decodes BGR; pass channels="BGR" so colors render accurately.
                    frame_placeholder.image(annotated_bgr, channels="BGR", use_container_width=True)

            elapsed = time.monotonic() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

        if client.last_error:
            status_box.error(client.last_error)
    finally:
        cap.release()
        client.close()


if __name__ == "__main__":
    main()
