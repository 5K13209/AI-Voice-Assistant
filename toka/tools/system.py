"""PC を操作するツール。エネらしさの本体。

アプリ起動は許可リスト方式にしてある。任意パスを os.startfile に渡す形は
作らない。LLM が生成した文字列がそのまま実行対象になる経路を残さないため。
"""

from __future__ import annotations

import logging
import os
import subprocess

from .. import config
from .registry import tool

log = logging.getLogger(__name__)


@tool(
    risk="confirm",
    params={"name": "アプリの通称。メモ帳、電卓、ブラウザ など"},
    confirm="{args} を開いていい？",
)
def open_app(name: str) -> str:
    """許可リストにあるアプリケーションを起動する。"""
    # 通称は表記揺れしやすいので、まず完全一致、次に部分一致で探す。
    target = config.ALLOWED_APPS.get(name)
    if target is None:
        lowered = name.lower()
        for alias, command in config.ALLOWED_APPS.items():
            if lowered in alias.lower() or alias.lower() in lowered:
                target = command
                name = alias
                break

    if target is None:
        return (
            f"「{name}」は起動を許可されていません。"
            f"使えるのは: {', '.join(config.ALLOWED_APPS)}"
        )

    try:
        if target.startswith(("http://", "https://", "ms-settings:")):
            os.startfile(target)
        else:
            # shell=False。target は許可リスト由来の固定文字列のみ。
            subprocess.Popen([target], shell=False)
    except OSError as exc:
        return f"{name} の起動に失敗しました: {exc}"

    log.info("アプリ起動: %s (%s)", name, target)
    return f"{name} を起動しました。"


@tool(risk="safe")
def get_active_window() -> str:
    """今ユーザーが前面に出しているウィンドウのタイトルを取得する。"""
    title = active_window_title()
    if not title:
        return "前面のウィンドウを取得できませんでした。"
    return f"前面のウィンドウ: {title}"


def active_window_title() -> str | None:
    """前面ウィンドウのタイトル。proactive.py からも使う。

    pygetwindow を使わず ctypes で直に叩く。依存を増やさずに済み、
    Windows では十分に安定している。
    """
    if os.name != "nt":
        return None

    import ctypes

    user32 = ctypes.windll.user32
    handle = user32.GetForegroundWindow()
    if not handle:
        return None
    length = user32.GetWindowTextLengthW(handle)
    if length <= 0:
        return None
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, buffer, length + 1)
    return buffer.value or None


@tool(risk="safe")
def read_clipboard() -> str:
    """クリップボードの中身を読む。"""
    try:
        import pyperclip

        text = pyperclip.paste()
    except Exception as exc:
        return f"クリップボードを読めませんでした: {exc}"

    if not text:
        return "クリップボードは空です。"
    # 長文をそのままコンテキストに載せない。
    if len(text) > 2000:
        return f"クリップボード（先頭2000字）:\n{text[:2000]}"
    return f"クリップボード:\n{text}"


@tool(
    risk="confirm",
    params={"text": "書き込む文字列"},
    confirm="クリップボードを書き換えていい？",
)
def write_clipboard(text: str) -> str:
    """クリップボードに文字列を書き込む。"""
    try:
        import pyperclip

        pyperclip.copy(text)
    except Exception as exc:
        return f"クリップボードに書けませんでした: {exc}"
    return "クリップボードにコピーしました。"


def _endpoint_volume():
    """既定の再生デバイスの音量インターフェースを返す。

    ツールは executor スレッドで走るので、COM をそのスレッドで初期化する
    必要がある。comtypes は暗黙に行うことがあるが、明示した方が確実。
    """
    import comtypes
    from pycaw.pycaw import AudioUtilities

    try:
        comtypes.CoInitialize()
    except OSError:
        # 既に別のモードで初期化済みならそのまま使う。
        pass

    return AudioUtilities.GetSpeakers().EndpointVolume


@tool(
    risk="safe",
    params={"percent": "音量。0 から 100"},
)
def set_volume(percent: int) -> str:
    """システムの音量を変更する。"""
    percent = max(0, min(100, int(percent)))
    try:
        _endpoint_volume().SetMasterVolumeLevelScalar(percent / 100.0, None)
    except ImportError:
        return "音量操作には pycaw が必要です（pip install pycaw）。"
    except Exception as exc:
        return f"音量を変更できませんでした: {exc}"

    log.info("音量を %d%% に変更", percent)
    return f"音量を{percent}%にしました。"


@tool(risk="safe")
def get_volume() -> str:
    """現在のシステム音量を取得する。"""
    try:
        current = round(_endpoint_volume().GetMasterVolumeLevelScalar() * 100)
    except ImportError:
        return "音量取得には pycaw が必要です（pip install pycaw）。"
    except Exception as exc:
        return f"音量を取得できませんでした: {exc}"
    return f"現在の音量は{current}%です。"
