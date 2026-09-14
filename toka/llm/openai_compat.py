"""OpenAI 互換 /v1/chat/completions を話すバックエンド。

これ 1 本で Ollama / LM Studio / Groq / Cerebras / OpenRouter を賄う。
いずれも同じエンドポイント形式を持つため、プロバイダごとにアダプタを
書く必要がない。差分は base_url・モデル名・rpm・認証ヘッダだけで、
それは factory.py がプリセットとして持つ。

依存は httpx のみ。google-genai が既に httpx を引いているので、
このモジュールのために新しい依存は増えない。

実装上の主要な落とし穴が 3 つある:

1. ストリーミングのツール呼び出しは**断片で届く**。関数名は最初のチャンク
   だけに入り、引数の JSON は数文字ずつに割れて後続チャンクへ散る。
   チャンクごとに json.loads しようとすると必ず失敗する。index ごとに
   連結し、完成してから一度だけ組み立てる（_ToolCallAccumulator）。
2. 構造化出力の対応度がプロバイダごとに違う。json_schema →
   json_object → プロンプトのみ、の 3 段で降格する。
3. 推論モデル（qwen3 など）は既定で思考を出す。Ollama は思考を content
   とは別の "reasoning" フィールドへ入れるので読み上げは汚れないが、
   トークンを思考に使い切って content が空になることがある。プリセット
   側で think=false を渡して止める（extra_body）。思考を content へ
   インラインで混ぜる実装もあるため、<think> の除去も併せて行う。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from .limiter import RateLimiter
from .types import BackendError, Delta, Message, ToolCall

log = logging.getLogger(__name__)

# 接続・読み取りのタイムアウト。読み取りは生成が長引くので緩めに取る。
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 180.0

# 時間を置けば直る見込みのステータス。router がフォールバック判断に使う。
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class _ToolCallAccumulator:
    """ストリーミングで断片化したツール呼び出しを組み立てる。

    OpenAI 互換のストリームでは、1 件のツール呼び出しが次のように割れて届く:

        {"index":0,"id":"call_x","function":{"name":"open_app","arguments":""}}
        {"index":0,"function":{"arguments":"{\\"na"}}
        {"index":0,"function":{"arguments":"me\\":\\"メモ帳\\"}"}}

    id と name は最初のチャンクにしか入らず、arguments だけが後続で伸びる。
    index が同一のものを 1 件として扱い、arguments を文字列連結してから
    最後に一度だけパースする。

    Ollama のように 1 チャンクで全部返す実装もあるため、arguments が
    最初から dict で来るケースも受け付ける。
    """

    def __init__(self) -> None:
        # index -> {"id": str, "name": str, "args": str | dict}
        self._parts: dict[int, dict[str, Any]] = {}
        self._order: list[int] = []

    def feed(self, raw_calls: Sequence[dict[str, Any]]) -> None:
        for raw in raw_calls:
            if not isinstance(raw, dict):
                continue
            # index が省略される実装があるので 0 に寄せる。
            index = raw.get("index")
            index = 0 if index is None else int(index)

            slot = self._parts.get(index)
            if slot is None:
                slot = {"id": "", "name": "", "args": ""}
                self._parts[index] = slot
                self._order.append(index)

            if raw.get("id"):
                slot["id"] = str(raw["id"])

            function = raw.get("function") or {}
            if not isinstance(function, dict):
                continue

            if function.get("name"):
                # 名前も分割して届く実装があるため連結する。通常は 1 回で入る。
                slot["name"] = str(slot["name"]) + str(function["name"])

            arguments = function.get("arguments")
            if arguments is None:
                continue
            if isinstance(arguments, dict):
                # Ollama など、最初からパース済みの dict を返す実装。
                slot["args"] = arguments
            elif isinstance(slot["args"], str):
                slot["args"] = slot["args"] + str(arguments)

    def finish(self) -> list[ToolCall]:
        """溜めたものを ToolCall のリストにする。届いた順を保つ。"""
        calls: list[ToolCall] = []
        for position, index in enumerate(self._order):
            slot = self._parts[index]
            name = str(slot["name"]).strip()
            if not name:
                # 名前が無いものは呼べない。壊れた断片として捨てる。
                log.warning("名前の無いツール呼び出しを破棄しました (index=%s)", index)
                continue

            calls.append(
                ToolCall(
                    # id を返さない実装があるので、無ければこちらで採番する。
                    # 履歴で tool 結果と突き合わせるのに必ず必要になる。
                    id=str(slot["id"]) or f"call_{position}_{name}",
                    name=name,
                    args=self._parse_args(slot["args"], name),
                )
            )
        return calls

    @staticmethod
    def _parse_args(raw: Any, name: str) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        text = str(raw).strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # 引数を復元できない。空で通すと llm.py 側が TypeError を捕まえて
            # 「引数が不正です」を LLM に返し、次のターンで直せる。
            log.warning("ツール %s の引数を解釈できません: %r", name, text[:200])
            return {}
        if not isinstance(parsed, dict):
            log.warning("ツール %s の引数が dict ではありません: %r", name, parsed)
            return {}
        return parsed

    def __bool__(self) -> bool:
        return bool(self._parts)


class ThinkFilter:
    """content に混ざった <think>...</think> を落とす。

    Ollama は思考を reasoning フィールドへ分離するので通常は不要だが、
    llama.cpp 直叩きや OpenRouter 経由の一部モデルは content にそのまま
    インラインで混ぜてくる。読み上げると「タグを音読する」ことになるので
    ここで止める。

    ストリーミングでは開始タグ・終了タグ自体がチャンク境界で割れるため、
    状態を持って処理する必要がある。タグらしき断片は、確定するまで
    保留する（部分一致を出力してしまわないように）。
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._inside = False
        self._pending = ""

    def feed(self, text: str) -> str:
        self._pending += text
        out = []

        while self._pending:
            if self._inside:
                index = self._pending.find(self.CLOSE)
                if index < 0:
                    # 終了タグの一部を掴んでいる可能性がある分だけ残す。
                    self._pending = self._keep_tail(self._pending, self.CLOSE)
                    break
                self._pending = self._pending[index + len(self.CLOSE) :]
                self._inside = False
                continue

            index = self._pending.find(self.OPEN)
            if index < 0:
                keep = self._keep_tail(self._pending, self.OPEN)
                emit = self._pending[: len(self._pending) - len(keep)]
                if emit:
                    out.append(emit)
                self._pending = keep
                break

            if index:
                out.append(self._pending[:index])
            self._pending = self._pending[index + len(self.OPEN) :]
            self._inside = True

        return "".join(out)

    def flush(self) -> str:
        """ストリーム終端。保留分を吐き出す（タグが未完なら捨てる）。"""
        rest = "" if self._inside else self._pending
        self._pending = ""
        return rest

    @staticmethod
    def _keep_tail(text: str, tag: str) -> str:
        """末尾が tag の接頭辞になっている分だけ返す。それ以外は確定とみなす。"""
        limit = min(len(tag) - 1, len(text))
        for size in range(limit, 0, -1):
            if tag.startswith(text[-size:]):
                return text[-size:]
        return ""


class LeakedToolCallFilter:
    """content に漏れたツール呼び出しを回収し、読み上げから隠す。

    ローカルモデルでは、ツール呼び出しの区切りトークンが正しく復号されず、
    サーバー側のパーサが呼び出しを取り出せないことがある。実測（Ollama +
    qwen2.5:14b、RDNA4）では 4 回に 3 回この状態になり、content に

        'คณะกรรม\\n{"name": "get_volume", "arguments": {}}\\nคณะกรรม'
        '_icall_\\n{"name": "get_volume", "arguments": {}}\\n_ick_'

    が出た。区切りが `<tool_call>` から別のトークン（タイ語や _icall_）へ
    化けているだけで、**JSON 自体は正しい**。ツール数やプロンプト長を変えても
    再現率は変わらなかったので、こちら側で回収するのが唯一の実用的な対処になる。

    放置すると 2 つ困る。ツールが実行されないこと、そして JSON が
    そのまま音声で読み上げられることである。

    先頭の疑わしい範囲だけを保留し、JSON を見つけたら区切りの化けた
    トークンごと捨てる。通常の文章だと分かった時点で保留を解いて素通しに
    切り替えるので、普通の応答の遅延にはならない。
    """

    # この長さまでに JSON が現れなければ、通常の文章として素通しする。
    # 漏れは必ず content の先頭に出るため、先頭だけ見れば足りる。
    PROBE_CHARS = 80

    # 回収後に残った断片を「化けた閉じ区切り」とみなす上限の長さ。
    # 実測の 'คณะกรรม' や '_ick_' はいずれもこれより短い。
    MARKER_MAX_CHARS = 40

    # 文の終わりとみなす文字。改行は含めない（閉じ区切りの直前が改行なので、
    # 含めると区切りを本文と誤判定する）。
    SENTENCE_END = "。！？!?."

    def __init__(self, tool_names: frozenset[str] | None = None) -> None:
        self._known = tool_names or frozenset()
        self._buffer = ""
        self._passthrough = False
        self._in_json = False
        self._json_start = 0
        self._post_salvage = False
        self.salvaged: list[ToolCall] = []

    def feed(self, text: str) -> str:
        if self._passthrough:
            return text

        self._buffer += text
        return self._drain()

    def _drain(self) -> str:
        if self._in_json:
            return self._continue_json()

        if self._post_salvage:
            # 回収済み。残りが本文なら流し、化けた閉じ区切りなら捨てる。
            if any(ch in self._buffer for ch in self.SENTENCE_END):
                self._post_salvage = False
                return self._release()
            return ""

        start = self._buffer.find("{")
        if start >= 0:
            # JSON らしきものが始まった。閉じるまで保留する。
            self._in_json = True
            self._json_start = start
            return self._continue_json()

        if len(self._buffer) >= self.PROBE_CHARS:
            # 先頭を十分見たが JSON は無い。通常の応答として素通しに切り替える。
            return self._release()

        # まだ判断できない。保留を続ける。
        return ""

    def _continue_json(self) -> str:
        """JSON の終わりを探す。閉じたら回収を試みる。

        深さは毎回 _json_start から数え直す。呼び出しを跨いで持ち越すと、
        同じ文字を二度数えて括弧が永久に閉じなくなる。
        """
        depth = 0
        for index in range(self._json_start, len(self._buffer)):
            char = self._buffer[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth != 0:
                    continue

                candidate = self._buffer[self._json_start : index + 1]
                if self._salvage(candidate):
                    # 開き側の化けた区切りごと捨てる。閉じ側は次の _drain で
                    # 判定する（本文が続く場合と区切りだけの場合があるため）。
                    self._buffer = self._buffer[index + 1 :]
                    self._in_json = False
                    self._post_salvage = True
                    return self._drain()

                # ツール呼び出しではなかった。そのまま流す。
                self._in_json = False
                return self._release()

        # まだ閉じていない。保留を続ける。
        return ""

    def _salvage(self, candidate: str) -> bool:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, dict):
            return False

        name = parsed.get("name")
        if not isinstance(name, str) or not name:
            return False
        # 未知の名前は回収しない。ユーザーが読み上げてほしい JSON を
        # 勝手にツール呼び出しへ化かさないため。
        if self._known and name not in self._known:
            return False
        if "arguments" not in parsed and "parameters" not in parsed:
            return False

        args = parsed.get("arguments", parsed.get("parameters")) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}

        log.info("content に漏れたツール呼び出しを回収しました: %s", name)
        self.salvaged.append(
            ToolCall(id=f"salvaged_{len(self.salvaged)}_{name}", name=name, args=args)
        )
        return True

    def _release(self) -> str:
        """保留をやめ、以降は素通しにする。"""
        self._passthrough = True
        out, self._buffer = self._buffer, ""
        return out

    def flush(self) -> str:
        """ストリーム終端。読み上げてはいけない残骸を落とす。"""
        if self._in_json:
            # 途中で切れた JSON。読み上げても意味がないので落とす。
            out = self._buffer[: self._json_start]
        elif self._post_salvage and len(self._buffer.strip()) <= self.MARKER_MAX_CHARS:
            # 回収した JSON の後ろに残った短い断片。句点も無いので
            # 化けた閉じ区切りと判断して落とす。
            out = ""
        else:
            out = self._buffer
        self._buffer = ""
        self._post_salvage = False
        return out


def to_openai_messages(
    messages: Sequence[Message], system: str = ""
) -> list[dict[str, Any]]:
    """中立の Message 列を OpenAI 形式へ変換する。

    Gemini と違い system は messages の先頭要素として送る。ツール結果は
    role="tool" + tool_call_id で返す（Gemini は role="user" の
    function_response なので、そこは gemini.py 側が受け持つ）。
    """
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for message in messages:
        if message.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.text,
                }
            )
            continue

        if message.role == "assistant" and message.tool_calls:
            entry: dict[str, Any] = {"role": "assistant"}
            # content は空でも省略せず null で入れる。省略すると
            # 400 を返すプロバイダがある。
            entry["content"] = message.text or None
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.args, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
            out.append(entry)
            continue

        out.append({"role": message.role, "content": message.text})

    return out


def to_openai_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """registry の中立スキーマを OpenAI の tools 形式へ包む。"""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            },
        }
        for tool in tools
    ]


class OpenAICompatBackend:
    """OpenAI 互換エンドポイントを話すバックエンド。"""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        rpm: int = 0,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.rpm = rpm
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._extra_headers = extra_headers or {}
        # プロバイダ固有の追加パラメータ（Ollama の think=false など）。
        self._extra_body = extra_body or {}
        self._limiter = RateLimiter(rpm)
        self._client: httpx.AsyncClient | None = None
        # 構造化出力の対応度は 1 回試せば分かる。降格結果を覚えて
        # 毎回 400 を踏まないようにする。
        self._structured_mode: str | None = None

    # =========================
    # ▼ HTTP
    # =========================

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._extra_headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT),
                headers=self._headers(),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _payload(
        self,
        messages: Sequence[Message],
        system: str,
        tools: Sequence[dict[str, Any]] | None,
        temperature: float,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **self._extra_body,
            "model": self.model,
            "messages": to_openai_messages(messages, system),
            "temperature": temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)
            payload["tool_choice"] = "auto"
        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response, body: str) -> None:
        status = response.status_code
        if status < 400:
            return

        retryable = status in RETRYABLE_STATUS
        detail = body.strip()[:400] or response.reason_phrase
        error = BackendError(
            f"HTTP {status}: {detail}", retryable=retryable, status=status
        )
        raise error

    def _note_retry_after(self, response: httpx.Response) -> None:
        """Retry-After を尊重する。自前のバックオフより正確なため。"""
        value = response.headers.get("retry-after")
        if not value:
            return
        try:
            self._limiter.penalize(float(value))
        except ValueError:
            # HTTP-date 形式。厳密に解釈する価値は薄いので既定値で待つ。
            self._limiter.penalize(30.0)

    # =========================
    # ▼ ストリーミング
    # =========================

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str = "",
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.8,
    ) -> AsyncIterator[Delta]:
        await self._limiter.acquire()
        client = self._ensure_client()
        payload = self._payload(messages, system, tools, temperature, stream=True)

        accumulator = _ToolCallAccumulator()
        think = ThinkFilter()
        leak = LeakedToolCallFilter(
            frozenset(t["name"] for t in tools) if tools else None
        )

        try:
            async with client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    self._note_retry_after(response)
                    self._raise_for_status(response, body)

                async for line in response.aiter_lines():
                    for delta in self._consume_line(
                        line, accumulator, think, leak
                    ):
                        yield delta
        except httpx.HTTPError as exc:
            # 接続断・タイムアウトは時間を置けば直る見込みとして扱う。
            raise BackendError(
                f"{self.name} への接続に失敗: {exc}", retryable=True
            ) from exc

        tail = leak.feed(think.flush()) + leak.flush()
        if tail:
            yield Delta(text=tail)

        # ストリームが終わってから、溜まったツール呼び出しを完成形で流す。
        calls = accumulator.finish()
        if not calls:
            # サーバー側が取り出せなかった分を、content から回収したもので補う。
            calls = leak.salvaged
        for call in calls:
            yield Delta(tool_call=call)

    def _consume_line(
        self,
        line: str,
        accumulator: _ToolCallAccumulator,
        think: ThinkFilter,
        leak: LeakedToolCallFilter,
    ) -> list[Delta]:
        """SSE の 1 行を処理して、流すべき Delta を返す。"""
        line = line.strip()
        if not line or line.startswith(":"):
            return []
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            return []

        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            log.debug("SSE の行を解釈できません: %r", line[:200])
            return []

        # プロバイダはエラーを 200 + ストリーム内で返すことがある。
        if isinstance(chunk.get("error"), dict):
            message = chunk["error"].get("message", "不明なエラー")
            raise BackendError(f"{self.name}: {message}", retryable=True)

        deltas: list[Delta] = []
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                continue

            raw_calls = delta.get("tool_calls")
            if raw_calls:
                accumulator.feed(raw_calls)

            content = delta.get("content")
            if content:
                visible = leak.feed(think.feed(str(content)))
                if visible:
                    deltas.append(Delta(text=visible))

        return deltas

    # =========================
    # ▼ 単発生成
    # =========================

    async def complete(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        messages = [Message(role="user", text=prompt)]

        for mode in self._structured_modes(schema):
            await self._limiter.acquire()
            payload = self._payload(messages, "", None, temperature, stream=False)
            self._apply_structured(payload, schema, mode)

            client = self._ensure_client()
            try:
                response = await client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                raise BackendError(
                    f"{self.name} への接続に失敗: {exc}", retryable=True
                ) from exc

            if response.status_code == 400 and mode != "none":
                # 構造化出力に未対応。1 段降格して再試行する。
                log.info(
                    "%s は %s に未対応のため降格します", self.name, mode
                )
                continue

            if response.status_code >= 400:
                self._note_retry_after(response)
                self._raise_for_status(response, response.text)

            # 対応が判明したので次回から同じモードを使う。ただし schema が
            # 無い呼び出しでは mode は常に "none" なので覚えてはいけない。
            # 覚えると、以降の構造化出力が恒久的に "none" に落ちる。
            if schema is not None:
                self._structured_mode = mode
            return self._extract_text(response.json())

        raise BackendError(f"{self.name}: 応答を取得できませんでした")

    def _structured_modes(self, schema: dict[str, Any] | None) -> list[str]:
        """試す順番を返す。判明済みならそれだけを返す。"""
        if schema is None:
            return ["none"]
        if self._structured_mode is not None:
            return [self._structured_mode]
        return ["json_schema", "json_object", "none"]

    @staticmethod
    def _apply_structured(
        payload: dict[str, Any], schema: dict[str, Any] | None, mode: str
    ) -> None:
        if schema is None or mode == "none":
            if schema is not None:
                # 構造化出力が使えないので、プロンプト側で頼む。
                payload["messages"][-1]["content"] += (
                    "\n\n次の JSON スキーマに従う JSON だけを出力してください。"
                    "前後に説明やコードフェンスを付けないでください。\n"
                    + json.dumps(schema, ensure_ascii=False)
                )
            return

        if mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
            payload["messages"][-1]["content"] += (
                "\n\nJSON だけを出力してください。"
            )
            return

        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": False,
            },
        }

    @staticmethod
    def _extract_text(body: dict[str, Any]) -> str:
        for choice in body.get("choices") or []:
            message = (choice or {}).get("message") or {}
            content = message.get("content")
            if content:
                # 非ストリーミングでも <think> が混ざる実装があるので通す。
                think = ThinkFilter()
                return (think.feed(str(content)) + think.flush()).strip()
        return ""

    # =========================
    # ▼ 疎通
    # =========================

    async def healthy(self) -> bool:
        client = self._ensure_client()
        try:
            response = await client.get(
                "/models", timeout=httpx.Timeout(5.0, connect=3.0)
            )
        except httpx.HTTPError:
            return False
        return response.status_code < 400

    def __repr__(self) -> str:  # pragma: no cover - デバッグ用
        return f"<OpenAICompatBackend {self.name} model={self.model} rpm={self.rpm}>"
