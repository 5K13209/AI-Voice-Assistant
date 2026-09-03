"""トーカ自身の内面に触れるツール。記憶への明示的な書き込みと、自発発話の制御。

set_timer と自発発話のオン・オフをツールにしてあるのが要点。「30分後に呼んで」
「今忙しいから黙ってて」が、特別扱いの命令ではなく普通の会話から実現される。
"""

from __future__ import annotations

import asyncio
import logging
import time

from .. import config
from ..events import EmotionChanged, TimerFired
from .context import CONTEXT
from .registry import tool

log = logging.getLogger(__name__)

# 起動中のタイマー。プロセスを落とすと消える（永続化はしない）。
_timers: dict[str, asyncio.Task] = {}

MAX_TIMERS = 10
MAX_TIMER_SECONDS = 12 * 3600


@tool(
    risk="safe",
    params={"fact": "覚えておく内容。一文で簡潔に"},
)
def remember(fact: str) -> str:
    """ユーザーについて判明した事実を長期記憶に書き込む。

    好み、習慣、予定、呼び方など、次の会話でも覚えていてほしいことに使う。
    ここに入れたものは毎回プロンプトに載るので、一時的な話題は入れない。
    """
    memory = CONTEXT.memory
    if memory is None:
        return "記憶が初期化されていません。"

    if memory.store_fact(fact):
        return f"覚えました: {fact}"
    return "それはもう覚えています。"


def feel(like: int = 0, fun: int = 0, anger: int = 0, sad: int = 0,
         trust: int = 0) -> str:
    """自分の感情が動いたときに、その変化量を記録する。

    値は「変化量」であって現在値ではない。動かない項目は 0 のままにする。
    日常会話の大半は感情が動かないので、呼ぶ必要はない。
    はっきり心が動いたときだけ呼ぶこと。

    応答と同じリクエストの中で呼ばれるので、API の消費が増えない。
    別モデルに投げ直す方式 (EMOTION_MODE="separate") より精度は落ちるが、
    無料枠の 5 リクエスト/分では 1 ターンに 2 回投げる余裕がない。
    """
    from ..services.emotion import apply, clamp_delta

    memory = CONTEXT.memory
    if memory is None:
        return "記憶が初期化されていません。"

    delta = clamp_delta(
        {"like": like, "fun": fun, "anger": anger, "sad": sad, "trust": trust}
    )
    if not any(delta.values()):
        return "変化なしとして記録しました。"

    updated = apply(memory.emotion, delta)
    memory.emotion = updated

    if CONTEXT.bus is not None:
        CONTEXT.bus.publish(EmotionChanged(emotion=updated, delta=delta))

    moved = {k: v for k, v in delta.items() if v}
    log.info("感情が動いた: %s -> %s", moved, updated)
    return f"感情を更新しました: {updated}"


# EMOTION_MODE が "separate"/"keyword" のときは登録しない。両方走らせると
# 同じ発言で感情が二重に動いてしまう。
if config.EMOTION_MODE == "tool":
    tool(
        risk="safe",
        params={
            "like": "好感度の変化。-5〜5",
            "fun": "楽しさの変化。-5〜5",
            "anger": "怒りの変化。-5〜5",
            "sad": "悲しさの変化。-5〜5",
            "trust": "信頼の変化。-5〜5",
        },
    )(feel)


@tool(risk="safe")
def recall_facts() -> str:
    """ユーザーについて今覚えていることを一覧する。"""
    memory = CONTEXT.memory
    if memory is None:
        return "記憶が初期化されていません。"

    text = memory.profile_text()
    return text or "まだ何も覚えていません。"


@tool(
    risk="safe",
    params={
        "seconds": "何秒後に知らせるか",
        "label": "何のためのタイマーか。読み上げる文言に使う",
    },
)
async def set_timer(seconds: int, label: str) -> str:
    """指定した秒数後に声をかけるタイマーを仕掛ける。"""
    bus = CONTEXT.bus
    if bus is None:
        return "タイマーを仕掛けられません。"

    seconds = int(seconds)
    if seconds <= 0:
        return "秒数は 1 以上で指定してください。"
    if seconds > MAX_TIMER_SECONDS:
        return "12時間より先のタイマーは仕掛けられません。"
    if len(_timers) >= MAX_TIMERS:
        return f"タイマーは同時に{MAX_TIMERS}件までです。"

    key = f"{label}-{time.time()}"

    async def fire() -> None:
        try:
            await asyncio.sleep(seconds)
            bus.publish(TimerFired(label=label))
        except asyncio.CancelledError:
            raise
        finally:
            _timers.pop(key, None)

    _timers[key] = asyncio.create_task(fire(), name=f"timer:{label}")

    minutes = seconds / 60
    when = f"{seconds}秒後" if seconds < 60 else f"{minutes:.0f}分後"
    log.info("タイマー設定: %s (%s)", label, when)
    return f"{when}に「{label}」で知らせます。"


@tool(risk="safe")
def list_timers() -> str:
    """仕掛かり中のタイマーを一覧する。"""
    if not _timers:
        return "仕掛かり中のタイマーはありません。"
    labels = [key.rsplit("-", 1)[0] for key in _timers]
    return "仕掛かり中: " + "、".join(labels)


@tool(risk="safe")
def cancel_timers() -> str:
    """仕掛かり中のタイマーをすべて取り消す。"""
    if not _timers:
        return "取り消すタイマーはありません。"
    count = len(_timers)
    for task in list(_timers.values()):
        task.cancel()
    _timers.clear()
    return f"タイマーを{count}件取り消しました。"


@tool(
    risk="safe",
    params={"enabled": "true で自分から話しかける、false で黙る"},
)
def set_proactive(enabled: bool) -> str:
    """自分から話しかけるかどうかを切り替える。

    「今忙しいから黙ってて」と言われたら false に、
    「もう話しかけていいよ」と言われたら true にする。
    """
    CONTEXT.proactive_enabled = bool(enabled)
    log.info("自発発話: %s", "有効" if enabled else "無効")
    if enabled:
        return "また自分から話しかけます。"
    return "呼ばれるまで黙っています。"


def shutdown() -> None:
    """終了時に仕掛かり中のタイマーを片付ける。"""
    for task in list(_timers.values()):
        task.cancel()
    _timers.clear()


__all__ = [
    "feel",
    "remember",
    "recall_facts",
    "set_timer",
    "list_timers",
    "cancel_timers",
    "set_proactive",
    "shutdown",
]
