"""ローカルとクラウドの併用を担う層。

3 つの役割にバックエンドを割り当てる:

    main      主応答。既定はローカル（回数無制限）
    sub       感情推定・エピソード要約・検索結果の要約。軽い仕事なので
              ローカルで十分。クラウドにすると無料枠を早く食い潰す
    fallback  main が落ちたときの逃げ先。任意

サービス層は router.main / router.sub を通してバックエンドに触る。
どのプロバイダが裏にいるかは知らない。

フォールバックは「1 回だけ」流す。何段も連鎖させると、どのプロバイダで
失敗したのかがログから追えなくなるうえ、待ち時間が積み上がって音声
アシスタントとしては手遅れになる。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from .base import ChatBackend
from .types import BackendError, Delta, Message

log = logging.getLogger(__name__)

# 一時的な失敗の再試行。常駐させる以上、これで応答を落としては困る。
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.5


class LLMRouter:
    def __init__(
        self,
        main: ChatBackend,
        *,
        sub: ChatBackend | None = None,
        fallback: ChatBackend | None = None,
    ) -> None:
        self.main = main
        # sub を省略したら main を兼用する。ローカルなら回数を気にしなくてよい。
        self.sub = sub or main
        self.fallback = fallback

    def describe(self) -> str:
        parts = [f"main={self.main.name}"]
        if self.sub is not self.main:
            parts.append(f"sub={self.sub.name}")
        if self.fallback is not None:
            parts.append(f"fallback={self.fallback.name}")
        return " / ".join(parts)

    # =========================
    # ▼ 主応答
    # =========================

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str = "",
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.8,
    ) -> AsyncIterator[Delta]:
        """main で流す。駄目なら再試行し、それでも駄目なら fallback へ回す。

        **一度でも差分を流したあとは、再試行も切り替えもしない。** 既に
        喋り始めたあとでやり直すと、同じ内容を二度読み上げることになる。
        旧実装も同じ理由で「接続確立までの失敗だけ再試行する」形にしていた。
        """
        last_error: BackendError | None = None

        for attempt in range(RETRY_ATTEMPTS):
            started = False
            try:
                async for delta in self.main.stream(
                    messages, system=system, tools=tools, temperature=temperature
                ):
                    started = True
                    yield delta
                return
            except BackendError as exc:
                if started:
                    # 流し始めていたので、やり直す選択肢はない。
                    raise
                last_error = exc
                if not exc.retryable or attempt == RETRY_ATTEMPTS - 1:
                    break
                delay = RETRY_BASE_DELAY * (2**attempt)
                log.warning(
                    "%s が失敗 (%s)。%.1f 秒後に再試行 (%d/%d)",
                    self.main.name,
                    exc,
                    delay,
                    attempt + 1,
                    RETRY_ATTEMPTS,
                )
                await asyncio.sleep(delay)

        assert last_error is not None
        if not self._can_fall_back(last_error):
            raise last_error

        assert self.fallback is not None
        log.warning(
            "%s を諦めて %s へ切り替えます: %s",
            self.main.name,
            self.fallback.name,
            last_error,
        )
        async for delta in self.fallback.stream(
            messages, system=system, tools=tools, temperature=temperature
        ):
            yield delta

    # =========================
    # ▼ 裏方の単発生成
    # =========================

    async def complete_sub(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """感情推定・要約など、裏方の生成。失敗したら fallback を試す。"""
        try:
            return await self.sub.complete(
                prompt, schema=schema, temperature=temperature
            )
        except BackendError as exc:
            if not self._can_fall_back(exc):
                raise
            log.warning("%s が失敗、fallback で再試行: %s", self.sub.name, exc)
            assert self.fallback is not None
            return await self.fallback.complete(
                prompt, schema=schema, temperature=temperature
            )

    def _can_fall_back(self, exc: BackendError) -> bool:
        if self.fallback is None:
            return False
        # 恒久的な失敗（400 や認証エラー）で逃げても同じ結果になるだけ。
        # 時間を置けば直る類のものだけ回す。
        return exc.retryable

    async def close(self) -> None:
        seen = set()
        for backend in (self.main, self.sub, self.fallback):
            if backend is None or id(backend) in seen:
                continue
            seen.add(id(backend))
            try:
                await backend.close()
            except Exception:  # pragma: no cover - 終了処理
                log.debug("%s の解放に失敗", backend.name, exc_info=True)
