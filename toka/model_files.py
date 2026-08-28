"""ASR モデルのファイル名解決。

sherpa-onnx が配布する Zipformer のファイル名にはエポック番号が入っている
（現行の ReazonSpeech k2-v2 は encoder-epoch-35-avg-1.int8.onnx）。この番号は
モデルの版が上がると変わるので、決め打ちすると取得スクリプトだけ通って
実行時に落ちる。glob で拾って、量子化版を優先する。

依存は pathlib だけにしてあるので、scripts/ からも軽く import できる。
"""

from __future__ import annotations

from pathlib import Path

MODEL_PARTS = ("encoder", "decoder", "joiner")

# 優先順。int8 が CPU では最速で、実用上の精度差もほぼ無い。
# decoder は int8 版が配布されない構成もあるため、素の .onnx まで落ちる。
_SUFFIX_PRIORITY = (".int8.onnx", ".onnx", ".fp16.onnx")


def resolve_model_file(model_dir: Path, part: str) -> Path:
    """`part` (encoder/decoder/joiner) に対応する onnx ファイルを 1 つ返す。"""
    candidates = sorted(model_dir.glob(f"{part}-*.onnx"))
    if not candidates:
        raise FileNotFoundError(
            f"{model_dir} に {part}-*.onnx が見つかりません。\n"
            "`python scripts/fetch_models.py --force` で取り直してください。"
        )

    for suffix in _SUFFIX_PRIORITY:
        for path in candidates:
            if path.name.endswith(suffix):
                return path

    return candidates[0]


def resolve_all(model_dir: Path) -> dict[str, Path]:
    """encoder/decoder/joiner/tokens をまとめて解決する。"""
    resolved = {part: resolve_model_file(model_dir, part) for part in MODEL_PARTS}

    tokens = model_dir / "tokens.txt"
    if not tokens.exists():
        raise FileNotFoundError(f"{tokens} が見つかりません。")
    resolved["tokens"] = tokens

    return resolved
