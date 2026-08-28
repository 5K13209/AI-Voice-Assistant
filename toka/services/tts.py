"""VOICEVOX による発話。文単位でパイプライン化し、途中で止められる。

旧実装は応答全文が返ってから合成し、sd.play() + sd.wait() で再生していた。
sd.wait() は一度始めると最後まで止まらないので、割り込みが不可能だった。
ここでは OutputStream に小さなチャンクを書き込むループにする。止めたければ
次のチャンクを書かなければよく、PLAYBACK_CHUNK 秒以内に必ず黙る。

合成と再生は別ワーカーにしてある。文 N を再生している間に文 N+1 を
合成できるので、2 文目以降の合成待ちが体感から消える。
"""

from __future__ import annotations

import asyncio
import io
import logging

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

from .. import config
from ..bus import EventBus
from ..events import (
    AssistantSentence,
    BargeIn,
    SpeechFinished,
    SpeechStarted,
)

log = logging.getLogger(__name__)


class VoicevoxError(RuntimeError):
    pass


def synthesize(text: str, speaker: int | None = None) -> tuple[np.ndarray, int]:
    """VOICEVOX で 1 文を合成する。ブロッキングなので executor から呼ぶこと。"""
    speaker = config.SPEAKER_ID if speaker is None else speaker
    try:
        query = requests.post(
            f"{config.VOICEVOX_URL}/audio_query",
            params={"text": text, "speaker": speaker},
            timeout=10,
        )
        query.raise_for_status()
        audio = requests.post(
            f"{config.VOICEVOX_URL}/synthesis",
            params={"speaker": speaker},
            json=query.json(),
            timeout=30,
        )
        audio.raise_for_status()
    except requests.RequestException as exc:
        raise VoicevoxError(
            f"VOICEVOX に接続できません ({config.VOICEVOX_URL})。起動していますか？"
        ) from exc

    data, rate = sf.read(io.BytesIO(audio.content), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, rate


def check_available() -> bool:
    """起動時の疎通確認。旧実装は失敗しても毎回黙って例外を握っていた。"""
    try:
        resp = requests.get(f"{config.VOICEVOX_URL}/version", timeout=3)
        resp.raise_for_status()
        log.info("VOICEVOX %s に接続しました", resp.text.strip().strip('"'))
        return True
    except requests.RequestException:
        log.error(
            "VOICEVOX (%s) に接続できません。起動してから実行してください。",
            config.VOICEVOX_URL,
        )
        return False


class TTSService:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._synth_queue: asyncio.Queue[str] = asyncio.Queue()
        # 2 文ぶんだけ先読みする。無制限にすると割り込み時に捨てる量が増える。
        self._play_queue: asyncio.Queue[tuple[str, np.ndarray, int]] = asyncio.Queue(
            maxsize=2
        )
        self._interrupt = asyncio.Event()
        self._stream: sd.OutputStream | None = None
        self._stream_rate = 0
        self._speaking = asyncio.Event()
        # 今まさに読み上げている文。割り込み判定でエコーを弾くのに使う。
        self._current_text = ""

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def current_text(self) -> str:
        return self._current_text

    def say(self, text: str) -> None:
        """1 文を発話キューに積む。すぐ返る。"""
        text = text.strip()
        if text:
            self._interrupt.clear()
            self._synth_queue.put_nowait(text)

    async def say_and_wait(self, text: str) -> None:
        """発話し、再生が終わる（か割り込まれる）まで待つ。起動時の挨拶など。"""
        self.say(text)
        await self.wait_until_idle()

    async def wait_until_idle(self) -> None:
        while (
            not self._synth_queue.empty()
            or not self._play_queue.empty()
            or self._speaking.is_set()
        ):
            await asyncio.sleep(0.05)

    def interrupt(self) -> None:
        """再生と待機中の文をすべて破棄する。BargeIn 受信時に呼ばれる。"""
        self._interrupt.set()
        for queue in (self._synth_queue, self._play_queue):
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    # =========================
    # ▼ ワーカー
    # =========================

    async def run(self) -> None:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._synth_worker(), name="tts-synth")
            tg.create_task(self._play_worker(), name="tts-play")
            tg.create_task(self._sentence_worker(), name="tts-sentences")
            tg.create_task(self._bargein_worker(), name="tts-bargein")

    async def _sentence_worker(self) -> None:
        async for event in self.bus.stream(AssistantSentence):
            self.say(event.text)

    async def _bargein_worker(self) -> None:
        async for _ in self.bus.stream(BargeIn):
            self.interrupt()

    async def _synth_worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            text = await self._synth_queue.get()
            if self._interrupt.is_set():
                continue
            try:
                data, rate = await loop.run_in_executor(None, synthesize, text)
            except VoicevoxError as exc:
                log.error("%s", exc)
                continue
            except Exception:
                log.exception("音声合成に失敗: %r", text)
                continue
            if self._interrupt.is_set():
                continue
            await self._play_queue.put((text, data, rate))

    async def _play_worker(self) -> None:
        while True:
            text, data, rate = await self._play_queue.get()
            if self._interrupt.is_set():
                continue
            await self._play(text, data, rate)

    def _ensure_stream(self, rate: int) -> sd.OutputStream:
        if self._stream is not None and self._stream_rate == rate:
            return self._stream
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._stream = sd.OutputStream(samplerate=rate, channels=1, dtype="float32")
        self._stream.start()
        self._stream_rate = rate
        return self._stream

    async def _play(self, text: str, data: np.ndarray, rate: int) -> None:
        stream = self._ensure_stream(rate)
        chunk = max(1, int(rate * config.PLAYBACK_CHUNK))

        self._speaking.set()
        self._current_text = text
        self.bus.publish(SpeechStarted(text=text))
        log.info("トーカ: %s", text)

        interrupted = False
        try:
            for start in range(0, len(data), chunk):
                if self._interrupt.is_set():
                    interrupted = True
                    break
                block = data[start : start + chunk]
                # write は書き込めるまでブロックするので executor に逃がす。
                # ここでループを止めると、割り込み判定自体が遅れる。
                await asyncio.get_running_loop().run_in_executor(
                    None, stream.write, block
                )
        except Exception:
            log.exception("再生に失敗")
        finally:
            self._speaking.clear()
            self._current_text = ""

        if interrupted:
            log.debug("再生を中断: %r", text)
        else:
            self.bus.publish(SpeechFinished(text=text))

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
