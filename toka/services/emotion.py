"""感情の更新。

旧実装はキーワードの単純加点だった。「好き」を含めば加点するので
「別に好きじゃない」でも上がるし、皮肉も文脈も読めない。応答生成のあとに
軽量モデルへ 1 回投げて、差分を自己申告させる形に変える。

本流のレイテンシには乗らない（応答を喋り始めたあとに走らせる）ので、
少し遅くても構わない。API が落ちたときのためにキーワード規則は
フォールバックとして残してある。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .. import config
from ..bus import EventBus
from ..events import EmotionChanged

log = logging.getLogger(__name__)

KEYS = ("like", "fun", "anger", "sad", "trust")

# 標準 JSON Schema（型名は小文字）。Gemini が要求する大文字への変換は
# toka/llm/gemini.py が行う。ここでプロバイダ固有の形にしてはいけない。
_SCHEMA = {
    "type": "object",
    "properties": {
        key: {
            "type": "integer",
            "description": f"{key} の変化量。-5 から 5。変化なしなら 0",
        }
        for key in KEYS
    },
    "required": list(KEYS),
}

_PROMPT = """あなたはAIアシスタント「トーカ」の感情を評価する係です。
以下のやり取りを読んで、トーカの感情が各項目どれだけ動いたかを答えてください。

判断の基準:
- 褒められた、感謝された、頼られた -> 好感度と信頼が上がる
- 楽しい話題、冗談が通じた -> 楽しさが上がる
- 否定された、雑に扱われた、無視された -> 怒りが上がり信頼が下がる
- 拒絶された、悲しい話題 -> 悲しさが上がる
- 皮肉や否定表現に注意する。「好きじゃない」は好意ではありません。
- 特筆すべき変化がなければ、すべて 0 にしてください。日常会話は大半が 0 です。

【ユーザーの発言】
{user}

【トーカの応答】
{assistant}
"""


class KeywordFallback:
    """API が使えないときの保険。旧 logic_utils.Emotion 相当。"""

    RULES = (
        (("好き", "ありがとう", "すごい", "最高"), {"like": 3, "trust": 2}),
        (("面白い", "楽しい", "興味深い", "草"), {"fun": 4}),
        (("キモイ", "下手", "変", "おかしい", "ばか", "うるさい"),
         {"anger": 5, "trust": -3}),
        (("残念", "がっかり", "失望", "嫌い"), {"sad": 4, "like": -2}),
    )

    @classmethod
    def delta(cls, text: str) -> dict[str, int]:
        result = dict.fromkeys(KEYS, 0)
        for words, changes in cls.RULES:
            if any(word in text for word in words):
                for key, value in changes.items():
                    result[key] += value
        return result


def clamp_delta(raw: dict[str, Any]) -> dict[str, int]:
    """LLM が極端な値を返しても暴れないように上限をかける。"""
    limit = config.EMOTION_MAX_DELTA
    delta = {}
    for key in KEYS:
        try:
            value = int(raw.get(key, 0))
        except (TypeError, ValueError):
            value = 0
        delta[key] = max(-limit, min(limit, value))
    return delta


def apply(emotion: dict[str, int], delta: dict[str, int]) -> dict[str, int]:
    """差分を適用して 0..100 に収める。元の dict は変更しない。"""
    updated = dict(emotion)
    for key in KEYS:
        updated[key] = max(0, min(100, updated.get(key, 50) + delta.get(key, 0)))
    return updated


class EmotionService:
    def __init__(self, bus: EventBus, router, memory) -> None:
        self.bus = bus
        self.router = router
        self.memory = memory

    async def update(self, user_text: str, assistant_text: str) -> None:
        delta = await self._estimate(user_text, assistant_text)

        if not any(delta.values()):
            return

        current = self.memory.emotion
        updated = apply(current, delta)
        self.memory.emotion = updated

        moved = {k: v for k, v in delta.items() if v}
        log.info("感情が動いた: %s -> %s", moved, updated)
        self.bus.publish(EmotionChanged(emotion=updated, delta=delta))

    async def _estimate(self, user_text: str, assistant_text: str) -> dict[str, int]:
        if config.EMOTION_MODE == "keyword":
            return clamp_delta(KeywordFallback.delta(user_text))

        try:
            raw = await self.router.complete_sub(
                _PROMPT.format(user=user_text, assistant=assistant_text),
                schema=_SCHEMA,
                temperature=0.0,
            )
        except Exception as exc:
            log.debug("感情推定に失敗、キーワード規則にフォールバック: %s", exc)
            return clamp_delta(KeywordFallback.delta(user_text))

        import json

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            log.debug("感情推定の応答を解釈できず、キーワード規則にフォールバック")
            return clamp_delta(KeywordFallback.delta(user_text))

        if not isinstance(parsed, dict):
            return clamp_delta(KeywordFallback.delta(user_text))
        return clamp_delta(parsed)


class EpisodeSummarizer:
    """一定ターンごとに会話を要約してエピソード記憶にする。

    生ログを 1 件ずつベクトル化すると、「ユーザー: うん」のような断片が
    大量に混ざって想起の邪魔になる。塊にして要約したものを引く。
    """

    PROMPT = """次の会話を、あとから思い出せるように3文以内で要約してください。
話題と、ユーザーについて分かったことを中心に。挨拶や相槌は省いてください。

{transcript}
"""

    def __init__(self, router, memory) -> None:
        self.router = router
        self.memory = memory
        self._lock = asyncio.Lock()

    async def maybe_summarize(self) -> None:
        if not self.memory.should_summarize():
            return

        # 要約中に次のターンが重なっても二重に走らせない。
        if self._lock.locked():
            return

        async with self._lock:
            transcript = self.memory.transcript_for_summary()
            if not transcript.strip():
                return

            try:
                summary = (
                    await self.router.complete_sub(
                        self.PROMPT.format(transcript=transcript),
                        temperature=0.3,
                    )
                ).strip()
            except Exception:
                log.exception("エピソードの要約に失敗")
                return

            if summary:
                await asyncio.get_running_loop().run_in_executor(
                    None, self.memory.store_episode, summary, self.memory.emotion
                )
