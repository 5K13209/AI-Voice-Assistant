"""入出力デバイスの選択。

旧 audio_handlers.auto_select_devices() は入力ループの中で return していたため、
最初のマイクが見つかった時点で関数を抜け、出力探索も sd.default.device への
代入も実行されていなかった。さらに認識入力は PyAudio 経由だったので、
仮に代入できていても STT には効かなかった。ここで両方を直す。
"""

from __future__ import annotations

import logging

import sounddevice as sd

from .config import (
    AUDIO_INPUT_DEVICE,
    AUDIO_INPUT_KEYWORDS,
    AUDIO_OUTPUT_DEVICE,
    AUDIO_OUTPUT_KEYWORDS,
)

log = logging.getLogger(__name__)


def _find(devices, channel_key: str, keywords: list[str]) -> int | None:
    for index, dev in enumerate(devices):
        if dev[channel_key] <= 0:
            continue
        name = dev["name"].lower()
        if any(kw.lower() in name for kw in keywords):
            return index
    return None


def select_devices() -> tuple[int | None, int | None]:
    """キーワードに一致するデバイスを探し、sd.default.device に反映する。

    見つからなければ None のままにして OS のデフォルトに委ねる。
    config で明示指定されていればそちらを優先する。
    """
    devices = sd.query_devices()

    input_id = AUDIO_INPUT_DEVICE
    if input_id is None:
        input_id = _find(devices, "max_input_channels", AUDIO_INPUT_KEYWORDS)

    output_id = AUDIO_OUTPUT_DEVICE
    if output_id is None:
        output_id = _find(devices, "max_output_channels", AUDIO_OUTPUT_KEYWORDS)

    if input_id is None:
        log.info("入力デバイスがキーワードに一致せず。OSデフォルトを使用します。")
    else:
        log.info("入力デバイス: [%d] %s", input_id, devices[input_id]["name"])

    if output_id is None:
        log.info("出力デバイスがキーワードに一致せず。OSデフォルトを使用します。")
    else:
        log.info("出力デバイス: [%d] %s", output_id, devices[output_id]["name"])

    sd.default.device = (input_id, output_id)
    return input_id, output_id


def describe_devices() -> str:
    """トラブルシュート用にデバイス一覧を文字列で返す。"""
    lines = []
    for index, dev in enumerate(sd.query_devices()):
        lines.append(
            f"[{index:2d}] in={dev['max_input_channels']:2d} "
            f"out={dev['max_output_channels']:2d}  {dev['name']}"
        )
    return "\n".join(lines)
