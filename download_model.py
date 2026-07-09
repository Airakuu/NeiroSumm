from __future__ import annotations

import sys
from pathlib import Path

import requests


BASE_URL = "https://huggingface.co/cointegrated/rut5-base-absum/resolve/main"
FILES = [
    "config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer_config.json",
    "model.safetensors",
]
TARGET_DIR = Path("models") / "rut5-base-absum"
CHUNK_SIZE = 1024 * 1024


def download_file(filename: str) -> None:
    url = f"{BASE_URL}/{filename}?download=true"
    destination = TARGET_DIR / filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    existing_size = destination.stat().st_size if destination.exists() else 0
    head_response = requests.head(url, allow_redirects=True, timeout=60)
    head_response.raise_for_status()
    remote_size = int(head_response.headers.get("Content-Length", 0))
    if remote_size and existing_size >= remote_size:
        print(f"{filename}: already downloaded")
        return

    headers = {"User-Agent": "SummarDownloader/0.1"}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        if response.status_code == 416:
            print(f"{filename}: already downloaded")
            return
        if response.status_code not in (200, 206):
            raise RuntimeError(f"Failed to download {filename}: HTTP {response.status_code}")

        total_size = remote_size or (existing_size + int(response.headers.get("Content-Length", 0)))
        mode = "ab" if response.status_code == 206 and existing_size > 0 else "wb"
        downloaded = existing_size if mode == "ab" else 0

        with destination.open(mode) as file_handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                file_handle.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = downloaded / total_size * 100
                    print(f"{filename}: {percent:.1f}% ({downloaded}/{total_size} bytes)")
                else:
                    print(f"{filename}: {downloaded} bytes")


def main() -> None:
    for filename in FILES:
        print(f"Downloading {filename}")
        download_file(filename)
    print(f"Model saved to {TARGET_DIR.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
