"""Gemini との対話。ストリーミング、ツール実行、履歴管理。

旧実装は client.chats.create() を使っていた。これを捨てて contents を自前で
持つ形にした理由が 4 つある。

1. chats.create は生成時の config を固定するので、system_instruction に
   埋め込んだ感情値が起動時のまま永久に更新されなかった。感情モデルを
   作り込んでも LLM には届いていなかった。
2. SDK の内部履歴と、毎ターン注入する想起記憶とで、同じ内容が二重に
   コンテキストへ乗っていた。
3. ツール応答 (function_response) を履歴に差し込む必要がある。
4. 割り込みで途中キャンセルした応答を、自分で整形して履歴に残す必要がある。

system_instruction は毎ターン組み直す。感情も記憶もここに載る。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from google.genai import types

from .. import config
from ..bus import EventBus
from ..events import (
    AssistantDelta,
    AssistantMessage,
    AssistantSentence,
    ThinkingStarted,
    ToolCalled,
    ToolConfirmationNeeded,
    ToolResult,
)
from ..tools.registry import ToolRegistry

log = logging.getLogger(__name__)

# 読み上げるので、Markdown の装飾記号は音にならない。落とす。
MARKDOWN_NOISE = re.compile(r"[*_`#>]|\[|\]|\(https?://[^)]*\)")

# 文の切れ目。ここまで溜まったら TTS に流す。
SENTENCE_END = "。！？!?\n"

# 句点が来ないまま伸び続ける応答（箇条書きなど）を、この長さで強制的に切る。
MAX_SENTENCE_CHARS = 60

# Gemini 側の一時的な障害。常駐させる以上、これで応答を落としては困る。
TRANSIENT_STATUS = (429, 500, 502, 503, 504)
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.5

AFFIRMATIVE = ("うん", "はい", "いい", "どうぞ", "おねが", "やって", "ok", "オーケ",
               "オッケ", "そう", "頼む", "たのむ", "ええ", "yes")
NEGATIVE = ("いや", "だめ", "ダメ", "やめ", "いらない", "しないで", "ちがう",
            "違う", "結構", "no")

PERSONA = """あなたは「トーカ」。ユーザーのPCの中に住んでいるAIです。

話し方:
- 基本は冷静で淡々としているが、感情の値に応じて温度が変わる。
- 音声で読み上げられるので、箇条書き・記号・URL・顔文字は使わない。
- 長々と説明しない。ひと息で言える長さを基本にする。
- 相手の発話は音声認識を通っているので、多少の誤字は文脈で補って読む。
  聞き取れていないと判断したら、素直に訊き返す。

感情の値について:
- 好感度・楽しさ・怒り・悲しさ・信頼の5つ。50が基準。
- 50を下回ればマイナス、超えればプラスの感情。最低0、最高100。
- 値をそのまま口に出すのではなく、その値に見合った態度で振る舞う。

ツールについて:
- PCを操作できる。頼まれたら実際に実行する。できるふりをしない。
- 実行結果は事実として扱う。失敗したら失敗したと言う。
- ユーザーについて長く覚えておくべきことを知ったら remember を使う。
- 心が動いたときだけ feel を使う。日常会話の大半では使わなくてよい。
"""


def retry_after(exc: Exception) -> float | None:
    """429 応答に含まれる retryDelay を秒で取り出す。

    無料枠の 429 は「41秒後に再試行してください」のように、待つべき時間を
    サーバーが教えてくれる。自前の指数バックオフより遥かに正確なので、
    あれば必ずそちらに従う。
    """
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        details = details.get("error", details).get("details", [])
    if not isinstance(details, list):
        return None

    for entry in details:
        if not isinstance(entry, dict):
            continue
        delay = entry.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                continue
    return None


class RateLimiter:
    """非同期トークンバケット。

    旧 logic_utils.can_send() は更新されないモジュール変数を見ていたので
    常に True を返し、呼び出し側も結果を捨てて sleep もしていなかった。
    README が説明していた 429 対策は実質存在しなかった。

    Gemini の無料枠は gemini-2.5-flash で 5 リクエスト/分。1 分あたりの
    上限からこちらで間隔を決め、サーバーに 429 を返させない側に倒す。
    サーバーが待ち時間を指定してきた場合は penalize() でそれに従う。
    """

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._last = 0.0
        self._until = 0.0
        self._lock = asyncio.Lock()

    def penalize(self, seconds: float) -> None:
        """サーバーに指定された時間だけ、次の送信を遅らせる。"""
        self._until = max(self._until, time.monotonic() + seconds)

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = max(
                self._min_interval - (now - self._last),
                self._until - now,
            )
            if wait > 0:
                log.info("レート制限のため %.1f 秒待機", wait)
                await asyncio.sleep(wait)
            self._last = time.monotonic()


class SentenceSplitter:
    """ストリーミング差分を、読み上げ可能な文に切り出す。"""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        sentences = []

        while True:
            index = next(
                (
                    i
                    for i, ch in enumerate(self._buffer)
                    if ch in SENTENCE_END
                ),
                None,
            )
            if index is None:
                # 句点が来ないまま長くなったら、読点で妥協して切る。
                # 切らないと最初の音が出るまで待たされ続ける。
                if len(self._buffer) >= MAX_SENTENCE_CHARS:
                    comma = self._buffer.rfind("、")
                    cut = comma + 1 if comma > 0 else MAX_SENTENCE_CHARS
                    sentences.append(self._buffer[:cut])
                    self._buffer = self._buffer[cut:]
                    continue
                break

            sentences.append(self._buffer[: index + 1])
            self._buffer = self._buffer[index + 1 :]

        return [s for s in (self._clean(x) for x in sentences) if s]

    def flush(self) -> str:
        rest = self._clean(self._buffer)
        self._buffer = ""
        return rest

    @staticmethod
    def _clean(text: str) -> str:
        return MARKDOWN_NOISE.sub("", text).strip()


class LLMService:
    def __init__(
        self,
        bus: EventBus,
        client: Any,
        memory,
        registry: ToolRegistry,
        tts,
    ) -> None:
        self.bus = bus
        self.client = client
        self.memory = memory
        self.registry = registry
        self.tts = tts

        self._history: list[types.Content] = []
        self._limiter = RateLimiter(config.MIN_REQUEST_INTERVAL)
        self._pending_confirmation: asyncio.Future[str] | None = None
        self._tools = self._build_tools()

    def _build_tools(self) -> list[types.Tool] | None:
        declarations = self.registry.declarations()
        if not declarations:
            return None
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(**d) for d in declarations
                ]
            )
        ]

    # =========================
    # ▼ プロンプト構築
    # =========================

    def _system_instruction(self, emotion: dict[str, int], recalled: str) -> str:
        parts = [PERSONA]

        parts.append(
            "\n【現在の感情】\n"
            f"好感度:{emotion['like']} 楽しさ:{emotion['fun']} "
            f"怒り:{emotion['anger']} 悲しさ:{emotion['sad']} 信頼:{emotion['trust']}"
        )

        profile = self.memory.profile_text()
        if profile:
            parts.append(f"\n【ユーザーについて覚えていること】\n{profile}")

        if recalled:
            parts.append(
                "\n【関連しそうな過去の記憶】\n"
                f"{recalled}\n"
                "（参考情報です。関係なければ無視してください）"
            )

        parts.append(f"\n【現在時刻】\n{time.strftime('%Y年%m月%d日 %H:%M')}")
        return "\n".join(parts)

    def _request_config(self, emotion: dict[str, int], recalled: str):
        return types.GenerateContentConfig(
            system_instruction=self._system_instruction(emotion, recalled),
            temperature=config.TEMPERATURE,
            tools=self._tools,
            # 手動でツールループを回すので、SDK の自動実行は止める。
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

    def _trim_history(self) -> None:
        limit = config.MAX_HISTORY_TURNS * 2
        if len(self._history) > limit:
            self._history = self._history[-limit:]

    # =========================
    # ▼ 確認待ち
    # =========================

    def is_awaiting_confirmation(self) -> bool:
        return (
            self._pending_confirmation is not None
            and not self._pending_confirmation.done()
        )

    def resolve_confirmation(self, text: str) -> None:
        if self.is_awaiting_confirmation():
            self._pending_confirmation.set_result(text)

    async def _confirm(self, tool, args: dict[str, Any]) -> bool:
        question = tool.question(args)
        self.bus.publish(
            ToolConfirmationNeeded(name=tool.name, args=args, question=question)
        )
        self.tts.say(question)

        loop = asyncio.get_running_loop()
        self._pending_confirmation = loop.create_future()
        try:
            answer = await asyncio.wait_for(self._pending_confirmation, timeout=25)
        except TimeoutError:
            log.info("確認がタイムアウトしたため %s を中止", tool.name)
            return False
        finally:
            self._pending_confirmation = None

        lowered = answer.lower()
        if any(word in lowered for word in NEGATIVE):
            return False
        return any(word in lowered for word in AFFIRMATIVE)

    # =========================
    # ▼ ツール実行
    # =========================

    async def _run_tool(self, call) -> dict[str, Any]:
        name = call.name
        args = dict(call.args or {})
        self.bus.publish(ToolCalled(name=name, args=args))

        tool = self.registry.get(name)
        if tool is None:
            result = f"{name} というツールはありません。"
            self.bus.publish(ToolResult(name=name, result=result, ok=False))
            return {"result": result}

        if tool.risk == "confirm" and not await self._confirm(tool, args):
            result = "ユーザーが許可しなかったので実行しませんでした。"
            self.bus.publish(ToolResult(name=name, result=result, ok=False))
            return {"result": result}

        try:
            result = await tool.call(args, config.TOOL_TIMEOUT)
            ok = True
        except TimeoutError:
            result = f"{name} が {config.TOOL_TIMEOUT} 秒以内に終わりませんでした。"
            ok = False
        except TypeError as exc:
            # LLM が引数を間違えた場合。そのまま返せば次のターンで直せる。
            result = f"引数が不正です: {exc}"
            ok = False
        except Exception as exc:
            log.exception("ツール %s の実行に失敗", name)
            result = f"実行時エラー: {exc}"
            ok = False

        self.bus.publish(ToolResult(name=name, result=result, ok=ok))
        return {"result": result}

    # =========================
    # ▼ 応答生成
    # =========================

    async def respond(
        self, user_text: str, trigger: str = "user", speak: bool = True
    ) -> str:
        """1 ターンぶんの応答を生成する。

        タスクとして起動され、割り込み時は外からキャンセルされる。
        キャンセルされた場合も、そこまでの発話を履歴に残して整合を保つ。

        speak=False にすると AssistantSentence を出さず、全文を返すだけに
        なる。自発発話でこれを使う。ストリーミングのまま喋らせると、
        「今は黙っておく」という判断が返ったときには手遅れになるため。
        """
        self.bus.publish(ThinkingStarted(trigger=trigger))

        emotion = self.memory.emotion
        recalled = await asyncio.get_running_loop().run_in_executor(
            None, self.memory.recall, user_text
        )
        request_config = self._request_config(emotion, recalled)

        self._history.append(
            types.Content(role="user", parts=[types.Part(text=user_text)])
        )
        self._trim_history()

        splitter = SentenceSplitter()
        full_text = ""
        interrupted = False

        try:
            for iteration in range(config.MAX_TOOL_ITERATIONS):
                await self._limiter.acquire()

                text, calls = await self._stream_once(request_config, splitter, speak)
                full_text += text

                if not calls:
                    break

                # ツール呼び出しは履歴に「モデルの発言」として残す必要がある。
                self._history.append(
                    types.Content(
                        role="model",
                        parts=(
                            ([types.Part(text=text)] if text else [])
                            + [types.Part(function_call=c) for c in calls]
                        ),
                    )
                )

                responses = []
                for call in calls:
                    outcome = await self._run_tool(call)
                    responses.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=call.name, response=outcome
                            )
                        )
                    )
                self._history.append(types.Content(role="user", parts=responses))
            else:
                log.warning("ツール呼び出しが上限に達したため打ち切り")

        except asyncio.CancelledError:
            interrupted = True
            raise
        finally:
            # splitter は full_text とは別に同じテキストを溜めているだけの
            # 読み上げ用ビュー。ここで得た端数を full_text に足すと、
            # 句点で終わらない応答の末尾が二重になる（SILENT が SILENTSILENT
            # になっていた）。読み上げに流すだけにする。
            tail = splitter.flush()
            if tail and not interrupted and speak:
                self.bus.publish(AssistantSentence(text=tail))

            # 記憶にも履歴にも装飾記号は要らない。読み上げ側と同じ整形をかける。
            full_text = SentenceSplitter._clean(full_text)

            if full_text:
                note = "（ここで遮られた）" if interrupted else ""
                self._history.append(
                    types.Content(
                        role="model", parts=[types.Part(text=full_text + note)]
                    )
                )
                self._trim_history()
                self.bus.publish(
                    AssistantMessage(text=full_text, interrupted=interrupted)
                )

        return full_text

    async def _stream_once(
        self, request_config, splitter, speak: bool = True
    ) -> tuple[str, list]:
        """1 回のストリーミング。テキストと function_call を返す。

        接続確立までの一時的な障害だけ再試行する。1 文でも読み上げ始めた
        あとに retry すると同じ内容を二度喋るので、そこから先は再試行しない。
        """
        text = ""
        calls: list = []

        stream = await self._open_stream(request_config)

        async for chunk in stream:
            for candidate in chunk.candidates or []:
                for part in (candidate.content.parts if candidate.content else []) or []:
                    if part.function_call:
                        calls.append(part.function_call)
                    if part.text:
                        text += part.text
                        self.bus.publish(AssistantDelta(text=part.text))
                        sentences = splitter.feed(part.text)
                        if speak:
                            for sentence in sentences:
                                self.bus.publish(AssistantSentence(text=sentence))

        return text, calls

    async def _open_stream(self, request_config):
        """ストリームを開く。過負荷や 429 は指数バックオフで再試行する。"""
        last_error: Exception | None = None

        for attempt in range(RETRY_ATTEMPTS):
            try:
                return await self.client.aio.models.generate_content_stream(
                    model=config.GEMINI_MODEL,
                    contents=self._history,
                    config=request_config,
                )
            except Exception as exc:
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if status not in TRANSIENT_STATUS:
                    raise
                last_error = exc

                # サーバーが待ち時間を指定してきたらそれに従う。無料枠の
                # 429 は「41秒後」のように具体的に返ってくる。
                delay = retry_after(exc)
                if delay is not None:
                    self._limiter.penalize(delay)
                else:
                    delay = RETRY_BASE_DELAY * (2**attempt)

                if attempt == RETRY_ATTEMPTS - 1:
                    break

                log.warning(
                    "Gemini が %s を返しました。%.1f 秒後に再試行 (%d/%d)",
                    status,
                    delay,
                    attempt + 1,
                    RETRY_ATTEMPTS,
                )
                await asyncio.sleep(delay)

        raise last_error

    def note_interruption(self) -> None:
        """割り込み時に、次のターンで文脈が途切れないようにする。"""
        if self._history and self._history[-1].role == "model":
            return
        self._history.append(
            types.Content(
                role="model",
                parts=[types.Part(text="（言いかけたところで遮られた）")],
            )
        )
