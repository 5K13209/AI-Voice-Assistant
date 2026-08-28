"""asyncio 上の pub/sub イベントバス。

サービスは publish するだけで、誰が聞いているかを知らない。購読側は型で
絞って待つ。これによって「画面出力を後から足す」が既存コードへの変更ゼロで
できるようになる。

購読には二通りある:

    # 1. 流れてくるものを順に処理する（ワーカー向け）
    async for ev in bus.stream(Utterance):
        ...

    # 2. 特定のイベントを 1 個だけ待つ（同期点向け）
    ev = await bus.wait_for(SpeechFinished, timeout=5)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TypeVar

from .events import Event

log = logging.getLogger(__name__)

E = TypeVar("E", bound=Event)

# 1 購読者あたりのバッファ。詰まったら古いものから捨てる。
# 音声フレームのような高頻度イベントで、遅い購読者がバスごと止めるのを防ぐ。
QUEUE_SIZE = 256


class Subscription:
    """1 購読者ぶんのキュー。async for で回すか、close() で解除する。"""

    def __init__(self, bus: EventBus, types: tuple[type[Event], ...]) -> None:
        self._bus = bus
        self._types = types
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._closed = False
        self.dropped = 0

    def matches(self, event: Event) -> bool:
        return not self._types or isinstance(event, self._types)

    def _offer(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # 一番古いものを捨てて新しいものを入れる。遅い購読者のせいで
            # publish 側がブロックする方が、取りこぼしより遥かに危険。
            self.dropped += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    async def get(self) -> Event:
        return await self._queue.get()

    def __aiter__(self) -> AsyncIterator[Event]:
        return self

    async def __anext__(self) -> Event:
        if self._closed:
            raise StopAsyncIteration
        return await self._queue.get()

    def close(self) -> None:
        self._closed = True
        self._bus._unsubscribe(self)

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class EventBus:
    def __init__(self) -> None:
        self._subs: list[Subscription] = []

    def subscribe(self, *types: type[Event]) -> Subscription:
        """指定した型のイベントを購読する。型を省略すると全イベント。"""
        sub = Subscription(self, types)
        self._subs.append(sub)
        return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)

    def publish(self, event: Event) -> None:
        """イベントを配る。同期メソッドなので、どのスレッドからでも
        呼べるわけではない点に注意（イベントループのスレッドから呼ぶこと）。
        別スレッドからは loop.call_soon_threadsafe(bus.publish, ev) を使う。"""
        log.debug("publish %s", type(event).__name__)
        for sub in list(self._subs):
            if sub.matches(event):
                sub._offer(event)

    async def stream(self, *types: type[Event]) -> AsyncIterator[Event]:
        """`async for ev in bus.stream(X, Y)` の形で回す。"""
        sub = self.subscribe(*types)
        try:
            while True:
                yield await sub.get()
        finally:
            sub.close()

    async def wait_for(
        self,
        *types: type[Event],
        timeout: float | None = None,
    ) -> Event | None:
        """指定型のイベントを 1 つ待つ。タイムアウトしたら None。"""
        sub = self.subscribe(*types)
        try:
            if timeout is None:
                return await sub.get()
            return await asyncio.wait_for(sub.get(), timeout)
        except TimeoutError:
            return None
        finally:
            sub.close()
