"""起動・配線・状態遷移。

旧 main.py の while True ループに相当する層だが、順番に呼び出す代わりに
サービスを並列に走らせてイベントで繋ぐ。どのサービスも他のサービスを
直接呼ばないので、画面出力を足すときはバスを購読するタスクを 1 本
追加するだけで済む。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from . import config
from .audio_devices import describe_devices, select_devices
from .bus import EventBus
from .events import (
    AssistantMessage,
    BargeIn,
    PartialTranscript,
    ProactiveImpulse,
    SpeakerRejected,
    State,
    StateChanged,
    UserMessage,
)
from .services.auth import AuthService, RegisterVoice, VoiceAuth
from .services.capture import AudioCapture
from .services.emotion import EmotionService, EpisodeSummarizer
from .services.llm import LLMService
from .services.memory import MemoryManager
from .services.proactive import ProactiveService
from .services.stt import STTService
from .services.tts import TTSService, check_available
from .tools import context as tool_context
from .tools.registry import load_all

log = logging.getLogger(__name__)

# 自発発話で、LLM が「今は黙る」と判断したときに返す合図。
SILENT_MARKER = "SILENT"

PROACTIVE_PROMPT = """（これはユーザーの発言ではなく、あなた自身が気づいたことです）

{context}

今このタイミングで声をかけるべきか判断してください。
かけるなら、ひと言だけ自然に話しかけてください。
不要だと思うなら {marker} とだけ返してください。
用が無いのに話しかけるのは避けてください。ほとんどの場合は {marker} で構いません。
"""


class Application:
    def __init__(self) -> None:
        self.bus = EventBus()
        self.state = State.IDLE
        self.memory = MemoryManager()

        self.capture = AudioCapture()
        self.tts = TTSService(self.bus)
        self.stt: STTService | None = None
        self.llm: LLMService | None = None
        self.emotion: EmotionService | None = None
        self.summarizer: EpisodeSummarizer | None = None
        self.proactive = ProactiveService(self.bus)
        self.auth_service: AuthService | None = None

        self._response_task: asyncio.Task | None = None
        self._last_user_text = ""

    # =========================
    # ▼ 状態
    # =========================

    def set_state(self, new: State) -> None:
        if new == self.state:
            return
        old, self.state = self.state, new
        log.debug("状態 %s -> %s", old.value, new.value)
        self.bus.publish(StateChanged(old=old, new=new))
        if self.stt is not None:
            self.stt.set_speaking(new is State.SPEAKING)

    # =========================
    # ▼ 起動
    # =========================

    def _make_client(self):
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit(
                "GEMINI_API_KEY が設定されていません。"
                ".env-example を .env にコピーしてキーを入れてください。"
            )
        return genai.Client(api_key=api_key)

    def _ensure_voice_enrolled(self) -> VoiceAuth | None:
        if not config.VOICE_AUTH_ENABLED:
            log.warning("話者照合は無効です（TOKA_VOICE_AUTH=0）")
            return None

        refs = self.memory.voice_refs
        missing = [r for r in refs if not os.path.exists(r)]
        if not refs or missing:
            if missing:
                log.info("登録済みの音声ファイルが見つからないため登録し直します")
            refs = RegisterVoice.register()
            self.memory.voice_refs = refs

        return VoiceAuth(refs)

    async def setup(self) -> None:
        select_devices()

        if not check_available():
            raise SystemExit(
                "VOICEVOX を起動してから実行してください。\n"
                f"接続先: {config.VOICEVOX_URL}"
            )

        client = self._make_client()
        registry = load_all()
        tool_context.bind(self.bus, self.memory, client)

        # モデルのロードはどれも数秒かかる。まとめて executor に逃がす。
        loop = asyncio.get_running_loop()
        log.info("モデルを読み込んでいます...")
        self.stt = await loop.run_in_executor(None, STTService, self.bus)
        voice_auth = await loop.run_in_executor(None, self._ensure_voice_enrolled)
        self.auth_service = AuthService(self.bus, voice_auth)

        self.llm = LLMService(self.bus, client, self.memory, registry, self.tts)
        self.emotion = EmotionService(self.bus, client, self.memory)
        self.summarizer = EpisodeSummarizer(client, self.memory)

        self.capture.start()
        log.info("感情: %s", self.memory.emotion)

    # =========================
    # ▼ ハンドラ
    # =========================

    async def _handle_user_messages(self) -> None:
        async for event in self.bus.stream(UserMessage):
            self._last_user_text = event.text

            # ツールの確認待ちなら、この発話は「はい／いいえ」の答え。
            # 新しいターンを始めずに、待っている側へ渡す。
            if self.llm.is_awaiting_confirmation():
                self.llm.resolve_confirmation(event.text)
                continue

            await self._cancel_response()
            self._response_task = asyncio.create_task(
                self._respond(event.text), name="respond"
            )

    async def _handle_rejections(self) -> None:
        async for event in self.bus.stream(SpeakerRejected):
            log.debug("棄却: %s (%.3f)", event.text, event.score)

    async def _handle_bargein(self) -> None:
        """喋っている最中の発話を割り込みとして扱う。

        確定した認識（＝声紋照合を通ったもの）を待っていると、言い終わって
        から 1 秒近く経ってしまい、割り込みではなく順番待ちになる。途中経過
        (PartialTranscript) で止める。ただし自分の声の回り込みで止まっては
        本末転倒なので、今喋っている文と照合してエコーを弾く。
        """
        async for event in self.bus.stream(PartialTranscript):
            if self.state is not State.SPEAKING and self._response_task is None:
                continue
            if self._looks_like_echo(event.text):
                log.debug("自分の声の回り込みと判断: %s", event.text)
                continue

            log.info("割り込み検出: %s", event.text)
            self.bus.publish(BargeIn())
            self.tts.interrupt()
            await self._cancel_response()
            self.set_state(State.LISTENING)

    def _looks_like_echo(self, partial: str) -> bool:
        """今 TTS が読み上げている文と重なっていればエコーとみなす。"""
        speaking = self.tts.current_text
        if not speaking or len(partial) < 3:
            return False
        normalized = partial.strip()
        return normalized in speaking or speaking[: len(normalized)] == normalized

    async def _cancel_response(self) -> None:
        task = self._response_task
        self._response_task = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        self.llm.note_interruption()

    async def _respond(self, user_text: str) -> None:
        self.set_state(State.THINKING)
        try:
            self.memory.store_event("user", user_text, self.memory.emotion)
            text = await self.llm.respond(user_text)
            if text:
                self.memory.store_event("assistant", text, self.memory.emotion)
                # 感情推定と要約は喋り終わりを待たせない。裏で走らせる。
                asyncio.create_task(self._post_turn(user_text, text))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("応答生成に失敗")
            self.tts.say("うまく考えがまとまらなかった。もう一度言ってくれる？")
        finally:
            self._response_task = None

    async def _post_turn(self, user_text: str, assistant_text: str) -> None:
        try:
            # "tool" の場合、感情は応答と同じリクエスト内で feel ツールが
            # 更新済み。ここで別途投げると API を二重に消費するうえ、
            # 同じ発言で感情が二重に動いてしまう。
            if config.EMOTION_MODE != "tool":
                await self.emotion.update(user_text, assistant_text)
            await self.summarizer.maybe_summarize()
        except Exception:
            log.exception("ターン後処理に失敗")

    async def _handle_proactive(self) -> None:
        async for event in self.bus.stream(ProactiveImpulse):
            # 会話中・発話中には割り込まない。衝動は捨てる（溜めない）。
            if self.state is not State.IDLE or self._response_task is not None:
                log.debug("会話中のため衝動 %s を見送り", event.kind)
                continue

            prompt = PROACTIVE_PROMPT.format(context=event.context, marker=SILENT_MARKER)
            self.set_state(State.THINKING)
            try:
                # speak=False。喋ってから「やっぱり黙る」はできない。
                text = await self.llm.respond(
                    prompt, trigger="proactive", speak=False
                )
            except Exception:
                log.exception("自発発話の生成に失敗")
                self.set_state(State.IDLE)
                continue

            cleaned = text.strip()
            if not cleaned or SILENT_MARKER in cleaned:
                log.info("自発発話は見送られた")
                self.set_state(State.IDLE)
                continue

            self.memory.store_event("assistant", cleaned, self.memory.emotion)
            self.tts.say(cleaned)

    async def _track_speaking_state(self) -> None:
        """TTS の状態を状態機械に反映する。"""
        while True:
            await asyncio.sleep(0.05)
            if self.tts.is_speaking:
                if self.state is not State.SPEAKING:
                    self.set_state(State.SPEAKING)
            elif self.state is State.SPEAKING:
                self.set_state(
                    State.THINKING if self._response_task else State.IDLE
                )

    async def _track_idle(self) -> None:
        async for _ in self.bus.stream(AssistantMessage):
            if not self.tts.is_speaking and self._response_task is None:
                self.set_state(State.IDLE)

    # =========================
    # ▼ 実行
    # =========================

    async def _greet(self) -> None:
        """起動の挨拶。TaskGroup の中で走らせる必要がある。

        TTS のワーカーが動き出す前に say_and_wait を呼ぶと、合成キューを
        誰も消費しないまま待ち続けて起動時にデッドロックする。
        """
        await self.tts.say_and_wait("起動完了。")
        self.set_state(State.IDLE)
        log.info("待機中。話しかけてください。（Ctrl+C で終了）")

    async def run(self) -> None:
        await self.setup()

        try:
            async with asyncio.TaskGroup() as tg:
                # サービスを先に立ち上げる。購読を張り終える前にイベントを
                # publish すると取りこぼす。
                tg.create_task(self.tts.run(), name="tts")
                tg.create_task(self.auth_service.run(), name="auth")
                tg.create_task(self._handle_user_messages(), name="dispatch")
                tg.create_task(self._handle_rejections(), name="rejections")
                tg.create_task(self._handle_bargein(), name="bargein")
                tg.create_task(self._handle_proactive(), name="proactive-dispatch")
                tg.create_task(self._track_speaking_state(), name="state")
                tg.create_task(self._track_idle(), name="idle")
                await asyncio.sleep(0.1)

                tg.create_task(self._greet(), name="greet")
                tg.create_task(self.stt.run(self.capture.frames()), name="stt")
                tg.create_task(self.proactive.run(), name="proactive")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        from .tools import inner

        log.info("終了処理中...")
        inner.shutdown()
        self.capture.stop()
        self.tts.close()
        self.memory.save()


def setup_logging(verbose: bool) -> None:
    # Windows のコンソールは既定が cp932 なので、日本語ログで落ちる。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # 依存ライブラリのログは黙らせる。
    for noisy in ("httpx", "urllib3", "chromadb", "sentence_transformers",
                  "speechbrain", "asyncio", "google_genai", "numba", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="toka", description="音声アシスタント トーカ")
    parser.add_argument(
        "--model",
        choices=["flash", "pro"],
        default="flash",
        help="Gemini のモデル（既定: flash）",
    )
    parser.add_argument(
        "--stt",
        choices=["sherpa", "whisper"],
        help="音声認識エンジン。既定は sherpa (ReazonSpeech)",
    )
    parser.add_argument("--no-auth", action="store_true", help="話者照合を無効にする")
    parser.add_argument("--no-proactive", action="store_true", help="自発発話を止める")
    parser.add_argument("--list-devices", action="store_true", help="音声デバイス一覧")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    load_dotenv()

    if args.list_devices:
        print(describe_devices())
        return 0

    config.GEMINI_MODEL = f"gemini-2.5-{args.model}"
    if args.stt:
        config.STT_ENGINE = args.stt
    if args.no_auth:
        config.VOICE_AUTH_ENABLED = False
    if args.no_proactive:
        config.PROACTIVE_ENABLED = False

    app = Application()
    try:
        await app.run()
    except KeyboardInterrupt:
        log.info("終了します")
    except SystemExit as exc:
        # setup() が前提条件を満たさずに落ちた場合。理由だけ出す。
        log.error("%s", exc)
        return 1
    except BaseExceptionGroup as group:
        # TaskGroup 内で落ちたサービス。旧 main.py の裸 except と違い、
        # どのサービスが何で落ちたかを必ず残す。
        for exc in group.exceptions:
            if isinstance(exc, KeyboardInterrupt):
                log.info("終了します")
                return 0
            log.error("サービスが停止しました: %s: %s", type(exc).__name__, exc)
            log.debug("詳細", exc_info=exc)
        return 1
    return 0


def run() -> int:
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        return 0
