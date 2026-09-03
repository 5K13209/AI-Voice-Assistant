"""STT に必要な ONNX モデルを models/ 以下に取得する。

sherpa-onnx は PyTorch を持たないので、モデルは HuggingFace ではなく
k2-fsa/sherpa-onnx の GitHub Release から直接落とす。初回のみ実行すればよい。

    python scripts/fetch_models.py
    python scripts/fetch_models.py --force   # 既存を消して取り直す
"""

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toka.model_files import MODEL_PARTS, resolve_model_file  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

# ReazonSpeech k2-v2 Zipformer (Apache-2.0)。日本語 ASR 本体。
ASR_ARCHIVE = "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17.tar.bz2"
ASR_DIR = MODELS_DIR / ASR_ARCHIVE.removesuffix(".tar.bz2")

# Silero VAD (MIT)。発話区間の切り出しに使う。
VAD_FILE = MODELS_DIR / "silero_vad.onnx"


def _report(done: int, block: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100, done * block * 100 // total)
    print(f"\r  {pct:3d}%  ({total / 1024 / 1024:.1f} MB)", end="", flush=True)


def download(url: str, dest: Path) -> None:
    print(f"取得中: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=_report)
    tmp.replace(dest)
    print(f"\r  完了 -> {dest}")


def fetch_asr(force: bool) -> None:
    if ASR_DIR.exists() and not force:
        print(f"ASRモデルは取得済み: {ASR_DIR}")
        return
    if ASR_DIR.exists():
        shutil.rmtree(ASR_DIR)

    archive = MODELS_DIR / ASR_ARCHIVE
    if not archive.exists() or force:
        download(f"{RELEASE}/{ASR_ARCHIVE}", archive)

    print("展開中...")
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(MODELS_DIR, filter="data")
    archive.unlink()
    print(f"  完了 -> {ASR_DIR}")


def fetch_vad(force: bool) -> None:
    if VAD_FILE.exists() and not force:
        print(f"VADモデルは取得済み: {VAD_FILE}")
        return
    download(f"{RELEASE}/silero_vad.onnx", VAD_FILE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="既存モデルを取り直す")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    fetch_asr(args.force)
    fetch_vad(args.force)

    # sherpa-onnx が実際に読むファイルが揃っているか確認する。
    # エポック番号はモデルの版で変わる（現行は epoch-35-avg-1）ので
    # 決め打ちせず glob で拾う。toka/services/stt.py も同じ探し方をする。
    try:
        found = {part: resolve_model_file(ASR_DIR, part) for part in MODEL_PARTS}
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        print("\n展開後のディレクトリ構成:", file=sys.stderr)
        for p in sorted(ASR_DIR.rglob("*")):
            print(f"  {p.relative_to(MODELS_DIR)}", file=sys.stderr)
        return 1

    if not (ASR_DIR / "tokens.txt").exists() or not VAD_FILE.exists():
        print("\ntokens.txt または silero_vad.onnx がありません。", file=sys.stderr)
        return 1

    print("\nすべて揃いました。")
    for part, path in found.items():
        print(f"  {part:8s} {path.name}")
    print(f"  {'vad':8s} {VAD_FILE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
