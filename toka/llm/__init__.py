"""LLM バックエンドの抽象化。

サービス層はここから出てくるものだけに依存し、どのプロバイダが裏にいるかを
知らない。プロバイダの差し替えは環境変数 1 つで済む。

    types.py          プロバイダ非依存の会話表現（Message / ToolCall / Delta）
    base.py           ChatBackend プロトコル
    limiter.py        レート制限。バックエンドごとに 1 つ
    openai_compat.py  Ollama / LM Studio / Groq / Cerebras / OpenRouter
    gemini.py         Gemini（固有の作法をここへ閉じ込める）
    factory.py        プリセットと生成
    router.py         main / sub / fallback の振り分け
"""

from .base import ChatBackend
from .factory import PRESETS, available, build, build_router, describe, list_models
from .router import LLMRouter
from .types import BackendError, Delta, Message, ToolCall

__all__ = [
    "PRESETS",
    "BackendError",
    "ChatBackend",
    "Delta",
    "LLMRouter",
    "Message",
    "ToolCall",
    "available",
    "build",
    "build_router",
    "describe",
    "list_models",
]
