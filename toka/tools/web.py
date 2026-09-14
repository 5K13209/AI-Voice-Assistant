"""Web 検索。

以前は Gemini の Google Search グラウンディングに投げていた。整った文章が
そのまま返るので読み上げに回しやすく、依存も増えないという利点があったが、
Gemini 固有の機能なのでプロバイダを替えると丸ごと失われる。

代わりに検索 API（Tavily）で素材を取り、要約は本体の LLM にやらせる
2 段構成にした。プロバイダに依存せず同じ動作になり、「読み上げ向けの
整った文章が返る」という利点も保てる。

Tavily を選んだ理由は無料枠がクレカ不要（月 1,000 credits）だから。
Brave Search API はクレカ認証が必要、DuckDuckGo の非公式ライブラリは
規約違反かつ HTML 変更で壊れるため、どちらも採らなかった。

スクレイピングはしない。生の HTML を読み上げに回せる形に整えるコストが
検索 API の無料枠より高くつく。
"""

from __future__ import annotations

import logging
import os

import httpx

from .context import CONTEXT
from .registry import tool

log = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_TIMEOUT = 15.0

# 要約に渡す検索結果の件数。多すぎるとコンテキストを食うだけで精度は上がらない。
MAX_RESULTS = 5

# 1 件あたりの本文の切り取り長。
MAX_SNIPPET_CHARS = 600

SUMMARY_PROMPT = """次の検索結果をもとに、質問へ日本語で簡潔に答えてください。

読み上げるので、箇条書き・記号・URL は使わないでください。
結果に答えが無ければ「分からなかった」と正直に言ってください。

【質問】
{query}

【検索結果】
{results}
"""


async def _search(query: str) -> list[dict[str, str]]:
    """Tavily を叩いて結果を返す。"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY が設定されていません。"
            "https://tavily.com で無料のキーを取得して .env に入れてください。"
        )

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": MAX_RESULTS,
        "search_depth": "basic",
        # Tavily 側の要約も貰っておく。検索 API が既に答えを持っている
        # ことがあり、そのときは LLM の要約が要らなくなる。
        "include_answer": True,
    }

    async with httpx.AsyncClient(timeout=TAVILY_TIMEOUT) as client:
        response = await client.post(TAVILY_URL, json=payload)
        if response.status_code == 401:
            raise RuntimeError("TAVILY_API_KEY が無効です。")
        response.raise_for_status()
        body = response.json()

    results = []
    if isinstance(body.get("answer"), str) and body["answer"].strip():
        results.append({"title": "検索エンジンの要約", "content": body["answer"]})

    for entry in body.get("results") or []:
        if not isinstance(entry, dict):
            continue
        content = str(entry.get("content") or "")[:MAX_SNIPPET_CHARS]
        if not content.strip():
            continue
        results.append(
            {"title": str(entry.get("title") or "無題"), "content": content}
        )

    return results


@tool(
    risk="safe",
    params={"query": "検索したい内容。自然文でよい"},
)
async def search_web(query: str) -> str:
    """Web を検索して、最新の情報を調べる。"""
    query = query.strip()
    if not query:
        return "検索語が空です。"

    try:
        results = await _search(query)
    except RuntimeError as exc:
        # 設定漏れ・キー無効。ユーザーに伝わる文言でそのまま返す。
        return str(exc)
    except httpx.HTTPError as exc:
        log.warning("Web検索に失敗: %s", exc)
        return f"検索できませんでした: {exc}"

    if not results:
        return f"「{query}」について、めぼしい結果が見つかりませんでした。"

    joined = "\n\n".join(
        f"{entry['title']}\n{entry['content']}" for entry in results
    )

    router = CONTEXT.llm_router
    if router is None:
        # 要約できないので素材をそのまま返す。読み上げには向かないが、
        # 何も返せないより情報がある方がよい。
        return joined[:1500]

    try:
        summary = await router.complete_sub(
            SUMMARY_PROMPT.format(query=query, results=joined),
            temperature=0.2,
        )
    except Exception as exc:
        log.warning("検索結果の要約に失敗: %s", exc)
        return joined[:1500]

    log.info("Web検索: %s", query)
    return summary.strip() or joined[:1500]
