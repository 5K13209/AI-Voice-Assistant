"""Gemini バックエンド。

旧 llm.py が直接持っていた Gemini 固有の作法を、すべてこのモジュールへ
閉じ込める。閉じ込める対象は 4 つある:

1. **ツール結果の返し方。** OpenAI 互換は role="tool" + tool_call_id だが、
   Gemini は role="user" の function_response で、しかも id ではなく
   **関数名**で紐付ける。id -> name の解決をここで行う。
2. **system の渡し方。** Gemini はリクエスト config のフィールドに載せる
   （messages の先頭要素ではない）。
3. **スキーマの型名。** Gemini は "OBJECT"/"INTEGER" と大文字を要求する。
   registry 側は標準 JSON Schema（小文字）を出すので、ここで変換する。
4. **エラー形。** 429 応答に含まれる retryDelay を秒で取り出す
   （Google API の RetryInfo 形式）。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from .limiter import RateLimiter
from .types import BackendError, Delta, Message, ToolCall

log = logging.getLogger(__name__)

TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})

# Gemini は JSON Schema の型名を大文字で要求する。
_TYPE_UPPER = {
    "object": "OBJECT",
    "array": "ARRAY",
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "null": "NULL",
}


def retry_after(exc: Exception) -> float | None:
    """429 応答に含まれる retryDelay を秒で取り出す。

    無料枠の 429 は「41秒後に再試行してください」のように、待つべき時間を
    サーバーが教えてくれる。自前の指数バックオフより遥かに正確なので、
    あれば必ずそちらに従う。

    ただし google-genai の APIError が .details を必ず持つ保証は無いので、
    取れなければ None を返して呼び出し側のバックオフに委ねる。
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


def to_gemini_schema(schema: Any) -> Any:
    """標準 JSON Schema を Gemini 形式（大文字の型名）へ再帰変換する。"""
    if isinstance(schema, list):
        return [to_gemini_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    converted: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            converted[key] = _TYPE_UPPER.get(value.lower(), value.upper())
        elif key in ("properties", "$defs", "definitions") and isinstance(value, dict):
            converted[key] = {k: to_gemini_schema(v) for k, v in value.items()}
        else:
            converted[key] = to_gemini_schema(value)
    return converted


def resolve_tool_name(messages: Sequence[Message], position: int) -> str:
    """messages[position] のツール結果が、どの関数の結果かを解決する。

    Gemini の function_response は id ではなく関数名で紐付くため、
    直前の assistant メッセージまで遡って tool_call_id が一致するものを探す。
    """
    target = messages[position].tool_call_id
    for index in range(position - 1, -1, -1):
        message = messages[index]
        for call in message.tool_calls:
            if call.id == target:
                return call.name
    log.warning("tool_call_id %s に対応する関数名が見つかりません", target)
    return "unknown_tool"


class GeminiBackend:
    """google-genai を使うバックエンド。"""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key: str,
        rpm: int = 5,
    ) -> None:
        from google import genai

        self.name = name
        self.model = model
        self.rpm = rpm
        self._client = genai.Client(api_key=api_key)
        self._limiter = RateLimiter(rpm)

    # =========================
    # ▼ 変換
    # =========================

    def _to_contents(self, messages: Sequence[Message]) -> list[Any]:
        from google.genai import types

        contents: list[Any] = []
        for position, message in enumerate(messages):
            if message.role == "tool":
                contents.append(
                    types.Content(
                        # Gemini はツール結果を "user" 側の発言として扱う。
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=resolve_tool_name(messages, position),
                                    response={"result": message.text},
                                )
                            )
                        ],
                    )
                )
                continue

            if message.role == "assistant":
                parts = []
                if message.text:
                    parts.append(types.Part(text=message.text))
                for call in message.tool_calls:
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=call.name, args=call.args
                            )
                        )
                    )
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
                continue

            contents.append(
                types.Content(role="user", parts=[types.Part(text=message.text)])
            )

        return contents

    def _to_tools(self, tools: Sequence[dict[str, Any]] | None) -> list[Any] | None:
        if not tools:
            return None
        from google.genai import types

        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=tool["name"],
                        description=tool.get("description", ""),
                        parameters=to_gemini_schema(
                            tool.get(
                                "parameters", {"type": "object", "properties": {}}
                            )
                        ),
                    )
                    for tool in tools
                ]
            )
        ]

    def _config(
        self,
        system: str,
        tools: Sequence[dict[str, Any]] | None,
        temperature: float,
    ) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            # Gemini は system をここに載せる。messages の先頭ではない。
            system_instruction=system or None,
            temperature=temperature,
            tools=self._to_tools(tools),
            # 手動でツールループを回すので、SDK の自動実行は止める。
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

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

        contents = self._to_contents(messages)
        request_config = self._config(system, tools, temperature)

        try:
            iterator = await self._client.aio.models.generate_content_stream(
                model=self.model, contents=contents, config=request_config
            )
        except Exception as exc:
            raise self._wrap(exc) from exc

        counter = 0
        try:
            async for chunk in iterator:
                for candidate in chunk.candidates or []:
                    parts = (
                        candidate.content.parts if candidate.content else []
                    ) or []
                    for part in parts:
                        if part.function_call:
                            call = part.function_call
                            counter += 1
                            yield Delta(
                                tool_call=ToolCall(
                                    # Gemini は id を返さないので採番する。
                                    id=f"gemini_{counter}_{call.name}",
                                    name=call.name,
                                    args=dict(call.args or {}),
                                )
                            )
                        if part.text:
                            yield Delta(text=part.text)
        except Exception as exc:
            raise self._wrap(exc) from exc

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
        from google.genai import types

        await self._limiter.acquire()

        if schema is None:
            request_config = types.GenerateContentConfig(temperature=temperature)
        else:
            request_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=to_gemini_schema(schema),
                temperature=temperature,
            )

        try:
            response = await self._client.aio.models.generate_content(
                model=self.model, contents=prompt, config=request_config
            )
        except Exception as exc:
            raise self._wrap(exc) from exc

        return (response.text or "").strip()

    # =========================
    # ▼ その他
    # =========================

    def _wrap(self, exc: Exception) -> BackendError:
        """SDK 例外を BackendError へ正規化し、待ち時間の指定を尊重する。"""
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        retryable = status in TRANSIENT_STATUS

        delay = retry_after(exc)
        if delay is not None:
            self._limiter.penalize(delay)

        return BackendError(
            f"{self.name}: {type(exc).__name__}: {exc}",
            retryable=retryable,
            status=status if isinstance(status, int) else None,
        )

    async def healthy(self) -> bool:
        # 疎通確認のためだけにトークンを消費したくない。キーの有無で判断する。
        return True

    async def close(self) -> None:
        return None

    def __repr__(self) -> str:  # pragma: no cover - デバッグ用
        return f"<GeminiBackend {self.name} model={self.model} rpm={self.rpm}>"
