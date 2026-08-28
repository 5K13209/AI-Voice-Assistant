"""自発発話。話しかけられなくても動く層。

エネの「勝手に出てくる」感はここが担う。ただし放っておくと単にうるさい
だけの存在になるので、抑制を三重にかけてある。

1. LLM に拒否権がある。衝動は「今声をかけるべきか」という問いとして渡り、
   大半は SILENT が返る。
2. クールダウン。反応が無ければ間隔を倍にしていく。
3. 静音時間帯と、会話で切れるキルスイッチ (set_proactive ツール)。

このモジュール自身は「衝動」を出すだけで、喋るかどうかは決めない。
判断は runtime が LLM に投げる。
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime

from .. import config
from ..bus import EventBus
from ..events import ProactiveImpulse, TimerFired, UserMessage
from ..tools.context import CONTEXT
from ..tools.system import active_window_title

log = logging.getLogger(__name__)


def in_quiet_hours(now: datetime | None = None) -> bool:
    """静音時間帯かどうか。(23, 8) のような日をまたぐ指定にも対応する。"""
    start, end = config.QUIET_HOURS
    hour = (now or datetime.now()).hour
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


class ProactiveService:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._last_user_at = time.monotonic()
        self._last_impulse_at = 0.0
        self._cooldown = float(config.PROACTIVE_COOLDOWN)
        self._window_title: str | None = None
        self._window_since = time.monotonic()
        self._window_notified = False

    # =========================
    # ▼ 抑制の判定
    # =========================

    def _allowed(self) -> bool:
        if not config.PROACTIVE_ENABLED or not CONTEXT.proactive_enabled:
            return False
        if in_quiet_hours():
            return False
        return time.monotonic() - self._last_impulse_at >= self._cooldown

    def _fire(self, kind: str, context: str) -> None:
        if not self._allowed():
            log.debug("衝動 %s は抑制された", kind)
            return
        self._last_impulse_at = time.monotonic()
        # 反応が無ければ間隔を倍にしていく。反応があれば on_user_message で戻す。
        self._cooldown = min(self._cooldown * 2, config.PROACTIVE_COOLDOWN_MAX)
        log.info("自発の衝動: %s (%s)", kind, context)
        self.bus.publish(ProactiveImpulse(kind=kind, context=context))

    def on_user_message(self) -> None:
        """ユーザーが話しかけてきたらタイマーとクールダウンをリセットする。"""
        self._last_user_at = time.monotonic()
        self._cooldown = float(config.PROACTIVE_COOLDOWN)
        self._window_notified = False

    # =========================
    # ▼ トリガ
    # =========================

    async def run(self) -> None:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._watch_user(), name="proactive-user")
            tg.create_task(self._idle_loop(), name="proactive-idle")
            tg.create_task(self._window_loop(), name="proactive-window")
            tg.create_task(self._timer_loop(), name="proactive-timer")

    async def _watch_user(self) -> None:
        async for _ in self.bus.stream(UserMessage):
            self.on_user_message()

    async def _idle_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            idle = time.monotonic() - self._last_user_at
            if idle >= config.PROACTIVE_IDLE_SECONDS:
                minutes = int(idle // 60)
                self._fire("idle", f"ユーザーが{minutes}分ほど何も話していない。")
                self._last_user_at = time.monotonic()

    async def _window_loop(self) -> None:
        """前面ウィンドウを見張る。「PCの中から見ている」感が一番出るところ。"""
        if not callable(active_window_title):
            return

        while True:
            await asyncio.sleep(config.WINDOW_POLL_INTERVAL)

            title = active_window_title()
            if title is None:
                continue

            if title != self._window_title:
                self._window_title = title
                self._window_since = time.monotonic()
                self._window_notified = False
                continue

            dwell = time.monotonic() - self._window_since
            if not self._window_notified and dwell >= config.WINDOW_DWELL_SECONDS:
                self._window_notified = True
                self._fire(
                    "window",
                    f"ユーザーは「{title}」を{int(dwell // 60)}分ほど開き続けている。",
                )

    async def _timer_loop(self) -> None:
        """set_timer で仕掛けたタイマー。これだけは抑制を通さず必ず喋る。"""
        async for event in self.bus.stream(TimerFired):
            log.info("タイマー発火: %s", event.label)
            self._last_impulse_at = time.monotonic()
            self.bus.publish(
                ProactiveImpulse(
                    kind="timer",
                    context=(
                        f"「{event.label}」で仕掛けたタイマーの時間になった。"
                        "必ず声をかけること。"
                    ),
                )
            )
