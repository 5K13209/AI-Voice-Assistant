"""バックエンドの共通インターフェース。

サービス層（llm.py / emotion.py / web.py）はこの Protocol にしか依存しない。
プロバイダを増やすときは、この 4 つを実装したクラスを 1 本足して factory に
登録するだけで済む。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, runtime_checkable

from .types import Delta, Message


@runtime_checkable
class ChatBackend(Protocol):
    """LLM バックエンド。

    name は表示・ログ用の識別子（"ollama:qwen3:14b" のような形）。
    rpm は 1 分あたりの許容リクエスト数で、0 は無制限（ローカル）を意味する。
    """

    name: str
    rpm: int

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str = "",
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.8,
    ) -> AsyncIterator[Delta]:
        """1 リクエストぶんのストリーミング。

        async generator を返すので、呼び出し側は `async for` で回す。
        tools は registry.declarations() が返す中立スキーマのリスト
        （{"name", "description", "parameters"}）。変換は adapter の責務。
        """
        ...

    async def complete(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """ストリーミングなしの単発生成。感情推定・要約・検索要約で使う。

        schema を渡した場合は JSON 文字列を返す努力をする。ただし
        バックエンドが構造化出力に対応していないことがあるので、
        「必ず JSON である」ことは保証しない。呼び出し側でパース失敗に
        備えること（emotion.py は KeywordFallback を持っている）。
        """
        ...

    async def healthy(self) -> bool:
        """疎通確認。ローカルバックエンドが落ちている場合の判定に使う。"""
        ...

    async def close(self) -> None:
        """HTTP クライアントなどの解放。"""
        ...
