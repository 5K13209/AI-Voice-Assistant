"""プロバイダのプリセットとバックエンド生成。

プロバイダを増やすときはここに 1 行足すだけで済む。OpenAI 互換なら
アダプタの実装は不要（openai_compat がそのまま使える）。

    TOKA_LLM_PROVIDER=ollama          主応答
    TOKA_LLM_SUB_PROVIDER=ollama      感情推定・要約・検索要約
    TOKA_LLM_FALLBACK_PROVIDER=       主応答が落ちたときの逃げ先（任意）

モデル名は TOKA_LLM_MODEL / TOKA_LLM_SUB_MODEL / TOKA_LLM_FALLBACK_MODEL
で個別に上書きできる。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from .base import ChatBackend
from .types import BackendError

log = logging.getLogger(__name__)

Kind = Literal["openai", "gemini"]


@dataclass(frozen=True)
class Preset:
    kind: Kind
    base_url: str
    # 既定モデル。クラウド側のモデル名は改廃されるので、404 が出たら
    # `python -m toka.llm --provider <name> --list-models` で確認して
    # TOKA_LLM_MODEL で上書きする。
    default_model: str
    rpm: int
    api_key_env: str | None = None
    # ローカルは鍵が不要。互換のため何か入れる必要がある実装向けの詰め物。
    dummy_key: str | None = None
    # プロバイダ固有の追加パラメータ。リクエスト本文へそのまま混ぜる。
    extra_body: dict[str, Any] = field(default_factory=dict)
    note: str = ""


PRESETS: dict[str, Preset] = {
    "ollama": Preset(
        kind="openai",
        base_url="http://127.0.0.1:11434/v1",
        # モデル選びで実測した落とし穴が 2 つある。
        #
        # 1. 推論モデル（qwen3 など）を選ばないこと。初トークンまで
        #    qwen3:14b が 9.98 秒、非推論モデルが 0.05〜0.22 秒だった。
        #    思考の分がそのまま応答の遅れになる。しかも Ollama の /v1 は
        #    think=false も /no_think も無視するので、モデル側で避けるしかない。
        # 2. qwen2.5:14b は Ollama 上でツール呼び出しのテンプレートが壊れて
        #    いる。区切りトークンが 'คณะกรรม' や '_icall_' に化け、呼び出しが
        #    解析されないまま生テキストが content へ漏れる。/v1 でもネイティブ
        #    /api/chat でも解析成功 0/4 だった（llama3.1:8b は 4/4）。
        #    日本語はやや自然だが、17 個のツールを持つこのアプリでは使えない。
        #
        # よって既定は llama3.1:8b。日本語はやや素朴だが、ツール呼び出しが
        # 確実に通る方が構成上重要である。
        default_model="llama3.1:8b",
        rpm=0,
        dummy_key="ollama",
        note="ローカル。回数無制限。Windows + RDNA4 では OLLAMA_VULKAN=1 が必要",
    ),
    "lmstudio": Preset(
        kind="openai",
        base_url="http://127.0.0.1:1234/v1",
        # ollama と同じ理由で非推論モデルを既定にする。
        default_model="qwen2.5-14b-instruct",
        rpm=0,
        dummy_key="lmstudio",
        note="ローカル。回数無制限。LM Studio でサーバーを開始しておくこと",
    ),
    "cerebras": Preset(
        kind="openai",
        base_url="https://api.cerebras.ai/v1",
        default_model="llama-3.3-70b",
        rpm=30,
        api_key_env="CEREBRAS_API_KEY",
        note="無料枠 30 RPM / 60〜100k TPM / 1M tok/日。クレカ不要",
    ),
    "groq": Preset(
        kind="openai",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        rpm=30,
        api_key_env="GROQ_API_KEY",
        note="無料枠 30 RPM / 6k TPM / 14,400 req/日。クレカ不要",
    ),
    "openrouter": Preset(
        kind="openai",
        base_url="https://openrouter.ai/api/v1",
        default_model="meta-llama/llama-3.3-70b-instruct:free",
        rpm=20,
        api_key_env="OPENROUTER_API_KEY",
        note="無料枠のあるモデルは :free 接尾辞つき",
    ),
    "gemini": Preset(
        kind="gemini",
        base_url="",
        default_model="gemini-2.5-flash",
        rpm=5,
        api_key_env="GEMINI_API_KEY",
        note="無料枠 5 RPM。比較用に残してある",
    ),
}


def available() -> list[str]:
    return sorted(PRESETS)


def describe() -> str:
    """--help や起動時の案内に出す一覧。"""
    lines = []
    for name in available():
        preset = PRESETS[name]
        rpm = "無制限" if preset.rpm <= 0 else f"{preset.rpm} RPM"
        lines.append(f"  {name:12} {rpm:8} {preset.note}")
    return "\n".join(lines)


def _env(prefix: str, suffix: str) -> str | None:
    value = os.getenv(f"{prefix}_{suffix}")
    return value.strip() if value and value.strip() else None


def build(
    provider: str,
    *,
    prefix: str = "TOKA_LLM",
    model: str | None = None,
) -> ChatBackend:
    """プロバイダ名からバックエンドを組む。

    prefix は環境変数の接頭辞で、role ごとに違うものを渡す
    （TOKA_LLM / TOKA_LLM_SUB / TOKA_LLM_FALLBACK）。
    """
    provider = provider.strip().lower()
    preset = PRESETS.get(provider)
    if preset is None:
        raise BackendError(
            f"未知のプロバイダです: {provider}\n"
            f"使えるのは: {', '.join(available())}"
        )

    chosen_model = model or _env(prefix, "MODEL") or preset.default_model

    api_key = None
    if preset.api_key_env:
        api_key = _env(prefix, "API_KEY") or os.getenv(preset.api_key_env)
        if not api_key:
            raise BackendError(
                f"{provider} を使うには {preset.api_key_env} が必要です。"
                f".env に設定してください。"
            )
    else:
        api_key = _env(prefix, "API_KEY") or preset.dummy_key

    if preset.kind == "gemini":
        from .gemini import GeminiBackend

        return GeminiBackend(
            name=f"{provider}:{chosen_model}",
            model=chosen_model,
            api_key=api_key or "",
            rpm=preset.rpm,
        )

    from .openai_compat import OpenAICompatBackend

    return OpenAICompatBackend(
        name=f"{provider}:{chosen_model}",
        base_url=_env(prefix, "BASE_URL") or preset.base_url,
        model=chosen_model,
        api_key=api_key,
        rpm=preset.rpm,
        extra_body=dict(preset.extra_body),
    )


def build_router(
    *,
    main: str | None = None,
    sub: str | None = None,
    fallback: str | None = None,
):
    """3 つの役割ぶんのバックエンドを組んで router を返す。

    main が指定されなければ TOKA_LLM_PROVIDER、それも無ければ ollama。
    sub を省略した場合は main を兼用する（ローカルなら回数を気にしなくてよい）。
    fallback は明示しない限り作らない。
    """
    from .router import LLMRouter

    main_provider = main or os.getenv("TOKA_LLM_PROVIDER") or "ollama"
    sub_provider = sub or os.getenv("TOKA_LLM_SUB_PROVIDER")
    fallback_provider = fallback or os.getenv("TOKA_LLM_FALLBACK_PROVIDER")

    main_backend = build(main_provider, prefix="TOKA_LLM")

    sub_backend = None
    if sub_provider and sub_provider != main_provider:
        sub_backend = build(sub_provider, prefix="TOKA_LLM_SUB")
    elif sub_provider:
        # 同じプロバイダでもモデルだけ替えたいことがある。
        sub_model = _env("TOKA_LLM_SUB", "MODEL")
        if sub_model:
            sub_backend = build(sub_provider, prefix="TOKA_LLM_SUB")

    fallback_backend = None
    if fallback_provider:
        try:
            fallback_backend = build(fallback_provider, prefix="TOKA_LLM_FALLBACK")
        except BackendError as exc:
            # フォールバックが組めないだけで起動を止める理由はない。
            log.warning("フォールバックを用意できませんでした: %s", exc)

    router = LLMRouter(
        main_backend, sub=sub_backend, fallback=fallback_backend
    )
    log.info("LLM: %s", router.describe())
    return router


async def list_models(provider: str) -> list[str]:
    """/models を叩いて使えるモデル名を返す。既定モデルが 404 のときに使う。"""
    preset = PRESETS.get(provider.strip().lower())
    if preset is None or preset.kind != "openai":
        return []

    backend = build(provider)
    try:
        client = backend._ensure_client()  # noqa: SLF001 - 診断用途
        response = await client.get("/models")
        response.raise_for_status()
        body: dict[str, Any] = response.json()
    except Exception as exc:
        log.error("モデル一覧を取得できませんでした: %s", exc)
        return []
    finally:
        await backend.close()

    return sorted(
        str(entry.get("id"))
        for entry in (body.get("data") or [])
        if isinstance(entry, dict) and entry.get("id")
    )
