"""Download the trained YOLO11s hand detection weights (best.pt) from GitHub Releases."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_REPO = "wyde1/realtime-yolo-hand-detection-app"
DEFAULT_TAG = "v1.0.0"
DEFAULT_ASSET = "best.pt"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "weights" / DEFAULT_ASSET


def build_release_url(repo: str = DEFAULT_REPO, tag: str = DEFAULT_TAG, asset: str = DEFAULT_ASSET) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{asset}"


def download_weights(url: str, output_path: Path, force: bool = False) -> Path:
    """Download model weights atomically to ``output_path``.

    Raises:
        RuntimeError: With actionable manual setup instructions if the release asset
            cannot be downloaded (e.g., HTTP 404 or network failure).
    """
    output_path = Path(output_path)
    if output_path.is_file() and not force:
        print(f"Weights already exist at '{output_path}'. Use --force to overwrite.")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    print(f"Downloading YOLO11s weights from:\n  {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "realtime-yolo-hand-detection-app/1.0"})

    try:
        with urllib.request.urlopen(request, timeout=60) as response, temp_path.open("wb") as fh:
            total_bytes = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            while True:
                chunk = response.read(1024 * 64)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if total_bytes > 0:
                    pct = (downloaded / total_bytes) * 100.0
                    print(
                        f"\rProgress: {downloaded / 1e6:.2f} / {total_bytes / 1e6:.2f} MB ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
            if total_bytes > 0:
                print()
    except urllib.error.HTTPError as exc:
        temp_path.unlink(missing_ok=True)
        if exc.code == 404:
            raise RuntimeError(
                f"Release asset not found (HTTP 404) at: {url}\n"
                "The GitHub Release may not be published yet, or the tag name differs.\n"
                "To set up the model manually:\n"
                f"  1. Obtain the trained YOLO11s checkpoint ('{DEFAULT_ASSET}'), or train it via "
                "training/train_yolo11_fingers.ipynb.\n"
                f"  2. Copy it manually to: {output_path}\n"
                "  3. Or point the server to another location with MODEL_PATH=/path/to/best.pt."
            ) from exc
        raise RuntimeError(
            f"HTTP {exc.code} while downloading weights from {url}.\n"
            f"Place '{DEFAULT_ASSET}' manually at '{output_path}' or set MODEL_PATH."
        ) from exc
    except urllib.error.URLError as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Network error while downloading weights from {url}: {exc.reason}\n"
            f"Place '{DEFAULT_ASSET}' manually at '{output_path}' or set MODEL_PATH."
        ) from exc
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    if not temp_path.is_file() or temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file was empty. Place '{DEFAULT_ASSET}' manually at '{output_path}'.")

    temp_path.replace(output_path)
    print(f"Saved weights to '{output_path}' ({output_path.stat().st_size / 1e6:.2f} MB).")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download trained YOLO11s weights from GitHub Releases.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repo slug (default: {DEFAULT_REPO}).")
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"Release tag (default: {DEFAULT_TAG}).")
    parser.add_argument("--url", default=None, help="Explicit download URL (overrides --repo/--tag).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Target path (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing weights file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    url = args.url or build_release_url(repo=args.repo, tag=args.tag)
    try:
        download_weights(url=url, output_path=args.output, force=args.force)
    except RuntimeError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
