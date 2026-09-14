"""自主レート制限。

旧 logic_utils.can_send() は更新されないモジュール変数を見ていたので常に
True を返し、呼び出し側も結果を捨てて sleep もしていなかった。README が
説明していた 429 対策は実質存在しなかった。

その後 llm.py に RateLimiter が置かれたが、通っていたのは主応答だけで、
感情推定・エピソード要約・Web検索の 3 経路は素通りしていた。1 ターンで
3 リクエスト飛ぶことがあり、5 RPM の無料枠では対策として機能しない。

そこでここへ移設し、**バックエンドごとに 1 インスタンス**を持たせて
stream() と complete() の両方が必ず acquire() を通る構造にした。素通りの
経路を作れないようにするのが目的である。
"""

from __future__ import annotations

import asyncio
import logging
import time

log = logging.getLogger(__name__)


class RateLimiter:
    """非同期トークンバケット。

    rpm=0 は無制限（ローカルバックエンド）。この場合 acquire() は即座に
    返り、ロックも取らない。ローカルで 13 秒待たされては本末転倒なので、
    無制限を特別扱いして経路ごと外す。

    サーバーが待ち時間を指定してきた場合は penalize() でそれに従う。
    自前の指数バックオフより遥かに正確なため。
    """

    def __init__(self, rpm: int, *, safety_margin: float = 1.1) -> None:
        self.rpm = rpm
        # 上限ぴったりだと境界で 429 を踏むので、1 割ほど余裕を持たせる。
        self._min_interval = 0.0 if rpm <= 0 else 60.0 / rpm * safety_margin
        self._last = 0.0
        self._until = 0.0
        self._lock = asyncio.Lock()

    @property
    def unlimited(self) -> bool:
        return self._min_interval <= 0.0

    def penalize(self, seconds: float) -> None:
        """サーバーに指定された時間だけ、次の送信を遅らせる。

        無制限バックエンドでも受け付ける。ローカルでも 503 を返すことは
        あるので、そのときだけは待つ必要がある。
        """
        if seconds <= 0:
            return
        self._until = max(self._until, time.monotonic() + seconds)

    async def acquire(self) -> None:
        if self.unlimited and self._until <= time.monotonic():
            return

        async with self._lock:
            now = time.monotonic()
            wait = max(
                self._min_interval - (now - self._last),
                self._until - now,
            )
            if wait > 0:
                log.info("レート制限のため %.1f 秒待機 (rpm=%d)", wait, self.rpm)
                await asyncio.sleep(wait)
            self._last = time.monotonic()
