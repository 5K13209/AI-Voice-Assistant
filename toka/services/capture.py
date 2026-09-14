"""マイクの常時キャプチャ。

全二重の土台。旧実装は recognizer.listen() で「録音を始めて、終わるまで
ブロックする」形だったので、喋っている間は原理的に耳が塞がっていた。
ここではストリームを開いたら二度と閉じず、20ms フレームを流し続ける。
録音の開始・停止という概念自体を持たない。

PortAudio のコールバックは専用スレッドから呼ばれるので、asyncio 側へは
loop.call_soon_threadsafe で渡す。コールバック内で重い処理をすると
音が途切れるため、ここでは詰め替えだけを行う。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np
import sounddevice as sd

from ..config import FRAME_SIZE, SAMPLE_RATE

log = logging.getLogger(__name__)

# フレームキューの上限。20ms × 200 = 4秒ぶん。
# これを超えるほど消費側が遅れているなら、古い音声はもう使い道がない。
QUEUE_FRAMES = 200


class AudioCapture:
    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=QUEUE_FRAMES)
        self._stream: sd.RawInputStream | None = None
        self.dropped_frames = 0

    # ---- PortAudio スレッドから呼ばれる ----
    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            # 入力オーバーフローは実際よく出る。落とすほどではないので記録のみ。
            log.debug("audio status: %s", status)
        # RawInputStream は CFFI バッファを再利用するのでコピー必須。
        samples = np.frombuffer(bytes(indata), dtype=np.int16)
        self._loop.call_soon_threadsafe(self._offer, samples)

    # ---- イベントループのスレッドで実行される ----
    def _offer(self, samples: np.ndarray) -> None:
        try:
            self._queue.put_nowait(samples)
        except asyncio.QueueFull:
            self.dropped_frames += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(samples)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    def start(self) -> None:
        if self._stream is not None:
            return
        self._loop = self._loop or asyncio.get_running_loop()
        try:
            self._stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=FRAME_SIZE,
                dtype="int16",
                channels=1,
                callback=self._callback,
            )
        except sd.PortAudioError as exc:
            # 既定の録音デバイスが無いと device=-1 の照会に失敗する。
            # 生のトレースバックだけ出ても原因が分からないので言い換える。
            raise SystemExit(
                f"マイクを開けませんでした: {exc}\n"
                "録音デバイスが接続・有効になっているか確認してください。\n"
                "  一覧: python -m toka --list-devices\n"
                "Windows のサウンド設定で既定の入力デバイスが選ばれていないと、"
                "デバイスが一覧に出ていても開けません。"
            ) from exc
        self._stream.start()
        log.info(
            "マイク常時キャプチャ開始 (%d Hz, %d サンプル/フレーム)",
            SAMPLE_RATE,
            FRAME_SIZE,
        )

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        log.info("マイクキャプチャ停止")

    async def frames(self) -> AsyncIterator[np.ndarray]:
        """[-1.0, 1.0] に正規化した float32 フレームを流し続ける。"""
        while True:
            samples = await self._queue.get()
            yield samples.astype(np.float32) / 32768.0

    def __enter__(self) -> AudioCapture:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
