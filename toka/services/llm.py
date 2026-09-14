"""LLM との対話。ストリーミング、ツール実行、履歴管理。

プロバイダには直接触らない。toka/llm/ の router を通すだけなので、Gemini でも
ローカルの Ollama でも、このファイルは同じまま動く。

履歴は自前で持つ。SDK の chats.create() を使わない理由が 4 つある。

1. chats.create は生成時の config を固定するので、system_instruction に
   埋め込んだ感情値が起動時のまま永久に更新されなかった。感情モデルを
   作り込んでも LLM には届いていなかった。
2. SDK の内部履歴と、毎ターン注入する想起記憶とで、同じ内容が二重に
   コンテキストへ乗っていた。
3. ツール応答を履歴に差し込む必要がある。
4. 割り込みで途中キャンセルした応答を、自分で整形して履歴に残す必要がある。

system は毎ターン組み直す。感情も記憶もここに載る。
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

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
from ..llm import BackendError, LLMRouter, Message, ToolCall
from . import persona
from .persona import Mode
from ..tools.registry import ToolRegistry

log = logging.getLogger(__name__)

# 読み上げるので、記号は音にならない。落とす。
# 波括弧と二重引用符も入れてある。小さいモデルは JSON の断片を応答へ
# 混ぜてくることがあり（実測で {"searchresults": } がそのまま出た）、
# 落とさないと記号を音読することになる。
MARKDOWN_NOISE = re.compile(r'[*_`#>{}"]|\[|\]|\(https?://[^)]*\)')

# 文の切れ目。ここまで溜まったら TTS に流す。
SENTENCE_END = "。！？!?\n"

# 句点が来ないまま伸び続ける応答（箇条書きなど）を、この長さで強制的に切る。
MAX_SENTENCE_CHARS = 60

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

絶対に守ること（モードによらず、例外なし）:
- 呼んでいないツールを呼んだことにしない。調べていないのに「調べた」と
  言わない。検索していないのに検索結果のように話さない。
- ツールが失敗したり使えなかったときは、その事実をそのまま伝える。
  もっともらしい答えで埋めてはいけない。
- 知らないことを知っているふうに話さない。分からないなら分からないと言う。
"""


def _failure_note(result: str) -> str:
    """失敗したツール結果に、捏造を禁じる指示を添える。

    システムプロンプトに「失敗したら失敗と言う」と書いても、小さいモデルは
    従わなかった。実測では、検索が「キーが未設定」で失敗した直後に
    「（検索結果）現在は115.35円でした」と数字を作ってきた。

    離れたところに置いた規則より、モデルが次に読むツール結果そのものへ
    書く方が効く。ここは正直度のダイヤルで緩めてよい部分ではないので、
    モードによらず常に添える。
    """
    return (
        f"[実行失敗] {result}\n"
        "この失敗をユーザーにそのまま伝えること。"
        "値や結果を推測・創作して答えてはいけない。"
    )


class SentenceSplitter:
    """ストリーミング差分を、読み上げ可能な文に切り出す。"""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        sentences = []

        while True:
            index = next(
                (i for i, ch in enumerate(self._buffer) if ch in SENTENCE_END),
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
        router: LLMRouter,
        memory,
        registry: ToolRegistry,
        tts,
    ) -> None:
        self.bus = bus
        self.router = router
        self.memory = memory
        self.registry = registry
        self.tts = tts

        self._history: list[Message] = []
        self._pending_confirmation: asyncio.Future[str] | None = None
        self._tools = self.registry.declarations() or None
        # 話し方のモード。runtime がユーザーの発話から切り替える。
        self.mode = Mode.CONVERSATION

    # =========================
    # ▼ プロンプト構築
    # =========================

    def _system(self, emotion: dict[str, int], recalled: str) -> str:
        parts = [PERSONA, persona.instructions(self.mode)]

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

        import time

        parts.append(f"\n【現在時刻】\n{time.strftime('%Y年%m月%d日 %H:%M')}")
        return "\n".join(parts)

    # =========================
    # ▼ 履歴の整合
    # =========================

    def _trim_history(self) -> None:
        """古い履歴を落とす。

        単純な末尾スライスだと、ツール呼び出しと結果の対を切り離してしまう。
        OpenAI 互換のプロバイダは「tool_calls を含む assistant の直後に
        対応する tool が並んでいない」履歴を 400 で弾くので、境界を直す。
        """
        limit = config.MAX_HISTORY_TURNS * 2
        if len(self._history) <= limit:
            return

        trimmed = self._history[-limit:]
        # 先頭に取り残された tool 結果は、対応する呼び出しを失っている。
        while trimmed and trimmed[0].role == "tool":
            trimmed.pop(0)
        self._history = trimmed

    def _heal_dangling_tool_calls(self) -> None:
        """結果の無いツール呼び出しに、打ち切りを示す結果を補う。

        割り込みでツール実行中にキャンセルされると、tool_calls を持つ
        assistant メッセージだけが履歴に残る。次のターンでそれを送ると
        OpenAI 互換のプロバイダは 400 を返す。ここで穴を埋める。
        """
        answered = {
            message.tool_call_id
            for message in self._history
            if message.role == "tool"
        }
        missing: list[Message] = []
        for message in self._history:
            for call in message.tool_calls:
                if call.id not in answered:
                    missing.append(
                        Message(
                            role="tool",
                            tool_call_id=call.id,
                            text="割り込みで中断されたため実行されませんでした。",
                        )
                    )
        if missing:
            log.debug("結果の無いツール呼び出し %d 件を補完", len(missing))
            self._history.extend(missing)

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

    async def _run_tool(self, call: ToolCall) -> str:
        name = call.name
        # 空のキーを落とす。引数を取らないツールに対して {"": ""} のような
        # 引数を付けてくるモデルがあり、そのまま渡すと
        # 「unexpected keyword argument ''」で必ず失敗する。
        args = {k: v for k, v in (call.args or {}).items() if k}
        self.bus.publish(ToolCalled(name=name, args=args))

        tool = self.registry.get(name)
        if tool is None:
            result = f"{name} というツールはありません。"
            self.bus.publish(ToolResult(name=name, result=result, ok=False))
            return result

        if tool.risk == "confirm" and not await self._confirm(tool, args):
            result = "ユーザーが許可しなかったので実行しませんでした。"
            self.bus.publish(ToolResult(name=name, result=result, ok=False))
            return result

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
        return result if ok else _failure_note(result)

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
        system = self._system(emotion, recalled)

        self._heal_dangling_tool_calls()
        self._history.append(Message(role="user", text=user_text))
        self._trim_history()

        splitter = SentenceSplitter()
        full_text = ""
        interrupted = False

        try:
            for _ in range(config.MAX_TOOL_ITERATIONS):
                text, calls = await self._stream_once(system, splitter, speak)
                full_text += text

                if not calls:
                    # 最終応答。ここで 1 度だけ履歴に入れる。
                    if text:
                        self._history.append(
                            Message(role="assistant", text=text)
                        )
                    break

                # ツール呼び出しは「モデルの発言」として残す必要がある。
                # 以前はこれと最終 full_text の二箇所に同じテキストを入れて
                # いたため、ツールを使ったターンで履歴が二重になっていた。
                self._history.append(
                    Message(role="assistant", text=text, tool_calls=calls)
                )

                for call in calls:
                    result = await self._run_tool(call)
                    self._history.append(
                        Message(
                            role="tool", tool_call_id=call.id, text=result
                        )
                    )
            else:
                log.warning("ツール呼び出しが上限に達したため打ち切り")

        except asyncio.CancelledError:
            interrupted = True
            raise
        except BackendError as exc:
            log.error("LLM が応答できませんでした: %s", exc)
            raise
        finally:
            # splitter は full_text とは別に同じテキストを溜めているだけの
            # 読み上げ用ビュー。ここで得た端数を full_text に足すと、
            # 句点で終わらない応答の末尾が二重になる。読み上げに流すだけ。
            tail = splitter.flush()
            if tail and not interrupted and speak:
                self.bus.publish(AssistantSentence(text=tail))

            # 記憶にも履歴にも装飾記号は要らない。読み上げ側と同じ整形をかける。
            full_text = SentenceSplitter._clean(full_text)

            self._heal_dangling_tool_calls()
            self._trim_history()

            if full_text:
                self.bus.publish(
                    AssistantMessage(text=full_text, interrupted=interrupted)
                )

        return full_text

    async def _stream_once(
        self, system: str, splitter: SentenceSplitter, speak: bool = True
    ) -> tuple[str, list[ToolCall]]:
        """1 回のストリーミング。テキストとツール呼び出しを返す。"""
        text = ""
        calls: list[ToolCall] = []

        async for delta in self.router.stream(
            self._history,
            system=system,
            tools=self._tools,
            temperature=config.TEMPERATURE,
        ):
            if delta.tool_call is not None:
                calls.append(delta.tool_call)
            if delta.text:
                text += delta.text
                self.bus.publish(AssistantDelta(text=delta.text))
                sentences = splitter.feed(delta.text)
                if speak:
                    for sentence in sentences:
                        self.bus.publish(AssistantSentence(text=sentence))

        return text, calls

    def note_interruption(self) -> None:
        """割り込み時に、次のターンで文脈が途切れないようにする。"""
        self._heal_dangling_tool_calls()
        if self._history and self._history[-1].role == "assistant":
            return
        self._history.append(
            Message(
                role="assistant", text="（言いかけたところで遮られた）"
            )
        )
