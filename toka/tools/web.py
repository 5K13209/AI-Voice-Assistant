"""Web 検索。

スクレイピングはせず、Gemini のグラウンディング検索に投げて要約を受け取る。
依存も API キーも増えず、返ってくるのが生の HTML ではなく整った文章なので、
そのまま読み上げに回せる。
"""

from __future__ import annotations

import logging

from .. import config
from .context import CONTEXT
from .registry import tool

log = logging.getLogger(__name__)


@tool(
    risk="safe",
    params={"query": "検索したい内容。自然文でよい"},
)
def search_web(query: str) -> str:
    """Web を検索して、最新の情報を調べる。"""
    client = CONTEXT.genai_client
    if client is None:
        return "検索機能が初期化されていません。"

    from google.genai import types

    try:
        response = client.models.generate_content(
            model=config.GEMINI_SUB_MODEL,
            contents=(
                f"次について調べて、日本語で簡潔にまとめてください。"
                f"読み上げるので箇条書きや記号は使わないでください。\n\n{query}"
            ),
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            ),
        )
    except Exception as exc:
        log.exception("Web検索に失敗")
        return f"検索できませんでした: {exc}"

    text = (response.text or "").strip()
    if not text:
        return "検索結果を取得できませんでした。"

    log.info("Web検索: %s", query)
    return text
