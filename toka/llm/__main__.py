"""バックエンドを横並びで叩く診断 CLI。

プロバイダを乗り換えるときに「何が動いて何が動かないか」を先に確定させる
ためのもの。本体を起動して音声で確かめるより遥かに速く切り分けられる。

    python -m toka.llm --list                        プリセット一覧
    python -m toka.llm --provider ollama --smoke     4 項目を叩く
    python -m toka.llm --provider groq --list-models モデル名を確認
    python -m toka.llm --provider ollama --chat      対話して感触を見る

--smoke で確認するのは次の 4 点:

    1. 単発生成      日本語で返るか
    2. ストリーミング 差分が届くか、最初の文字までどれくらいか
    3. ツール呼び出し 関数名と引数を正しく組み立てられるか
    4. 構造化出力     JSON スキーマに従うか（降格したならどの段か）

3 が最大の関門である。ローカルモデルは日本語でのツール呼び出しが
不安定なことがあり、トーカは 17 個のツールと、感情更新まで function
calling に載せている。ここが落ちるなら TOKA_EMOTION_MODE=separate へ
逃がす判断が必要になる。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time

from dotenv import load_dotenv

from . import factory
from .types import BackendError, Message

log = logging.getLogger("toka.llm.smoke")

# ツール呼び出しの検査用。実物の registry を使わないのは、pycaw など
# 環境依存の import を巻き込まずに LLM 側の能力だけを見たいため。
WEATHER_TOOL = {
    "name": "get_weather",
    "description": "指定した都市の現在の天気を取得する。",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "都市名"},
        },
        "required": ["city"],
    },
}

EMOTION_SCHEMA = {
    "type": "object",
    "properties": {
        "like": {"type": "integer", "description": "好感度の変化。-5〜5"},
        "fun": {"type": "integer", "description": "楽しさの変化。-5〜5"},
    },
    "required": ["like", "fun"],
}


def _ok(label: str, detail: str) -> None:
    print(f"[OK]   {label}: {detail}", flush=True)


def _fail(label: str, detail: str) -> None:
    print(f"[FAIL] {label}: {detail}", flush=True)


async def check_complete(backend) -> bool:
    try:
        started = time.monotonic()
        text = await backend.complete(
            "「起動確認」とだけ日本語で返してください。", temperature=0.0
        )
        elapsed = time.monotonic() - started
    except BackendError as exc:
        _fail("単発生成", str(exc))
        return False

    if not text:
        _fail("単発生成", "空の応答")
        return False
    _ok("単発生成", f"{elapsed:.2f}秒 / {text[:60]!r}")
    return True


async def check_stream(backend) -> bool:
    try:
        started = time.monotonic()
        first: float | None = None
        chunks = 0
        text = ""
        async for delta in backend.stream(
            [Message(role="user", text="自己紹介を2文でしてください。")],
            system="あなたは日本語で簡潔に話すアシスタントです。",
            temperature=0.7,
        ):
            if delta.text:
                if first is None:
                    first = time.monotonic() - started
                chunks += 1
                text += delta.text
        total = time.monotonic() - started
    except BackendError as exc:
        _fail("ストリーミング", str(exc))
        return False

    if not text.strip():
        _fail("ストリーミング", f"テキストが届きませんでした（{chunks} チャンク）")
        return False

    ttft = f"{first:.2f}秒" if first is not None else "不明"
    _ok(
        "ストリーミング",
        f"初トークン {ttft} / 全体 {total:.2f}秒 / {chunks} チャンク / "
        f"{len(text)} 文字",
    )
    print(f"       -> {text.strip()[:120]}", flush=True)
    return True


async def check_tool_call(backend) -> bool:
    """ここが乗り換え判断の本体。日本語でツールを正しく呼べるか。"""
    try:
        calls = []
        text = ""
        async for delta in backend.stream(
            [Message(role="user", text="東京の天気を調べて。")],
            system="ツールが使えます。必要なら必ず実際に呼び出してください。",
            tools=[WEATHER_TOOL],
            temperature=0.0,
        ):
            if delta.tool_call:
                calls.append(delta.tool_call)
            if delta.text:
                text += delta.text
    except BackendError as exc:
        _fail("ツール呼び出し", str(exc))
        return False

    if not calls:
        _fail(
            "ツール呼び出し",
            f"呼ばれませんでした（テキストのみ: {text.strip()[:80]!r}）",
        )
        return False

    call = calls[0]
    if call.name != "get_weather":
        _fail("ツール呼び出し", f"関数名が違います: {call.name}")
        return False
    if not isinstance(call.args, dict) or not call.args.get("city"):
        _fail("ツール呼び出し", f"引数を組み立てられていません: {call.args}")
        return False

    _ok("ツール呼び出し", f"{call.name}({call.args}) id={call.id}")

    # ツール結果を返して会話が継続できるかまで見る。ここが通らないと
    # 実際のツールループは回らない。
    try:
        followup = ""
        async for delta in backend.stream(
            [
                Message(role="user", text="東京の天気を調べて。"),
                Message(role="assistant", tool_calls=[call]),
                Message(
                    role="tool",
                    tool_call_id=call.id,
                    text="東京は晴れ、気温22度。",
                ),
            ],
            system="ツールの結果を事実として、日本語で簡潔に答えてください。",
            tools=[WEATHER_TOOL],
            temperature=0.0,
        ):
            if delta.text:
                followup += delta.text
    except BackendError as exc:
        _fail("ツール結果の返送", str(exc))
        return False

    if not followup.strip():
        _fail("ツール結果の返送", "結果を渡した後の応答が空")
        return False
    _ok("ツール結果の返送", followup.strip()[:80])
    return True


async def check_structured(backend) -> bool:
    try:
        raw = await backend.complete(
            "ユーザーに「ありがとう、助かった」と言われました。"
            "感情の変化量を答えてください。",
            schema=EMOTION_SCHEMA,
            temperature=0.0,
        )
    except BackendError as exc:
        _fail("構造化出力", str(exc))
        return False

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        _fail("構造化出力", f"JSON として読めません: {raw[:120]!r}")
        return False

    if not isinstance(parsed, dict) or "like" not in parsed:
        _fail("構造化出力", f"スキーマに合いません: {parsed}")
        return False

    mode = getattr(backend, "_structured_mode", None) or "native"
    _ok("構造化出力", f"{parsed} (mode={mode})")
    return True


async def run_smoke(provider: str, model: str | None) -> int:
    try:
        backend = factory.build(provider, model=model)
    except BackendError as exc:
        print(f"バックエンドを作れません: {exc}")
        return 1

    print(f"\n=== {backend.name} ===")
    rpm = "無制限" if backend.rpm <= 0 else f"{backend.rpm} RPM"
    print(f"レート上限: {rpm}\n")

    if not await backend.healthy():
        print(
            "疎通できません。ローカルなら次を確認してください:\n"
            "  - Ollama:    ollama serve が動いているか\n"
            "  - LM Studio: サーバーを開始しているか"
        )
        await backend.close()
        return 1

    try:
        results = [
            await check_complete(backend),
            await check_stream(backend),
            await check_tool_call(backend),
            await check_structured(backend),
        ]
    finally:
        await backend.close()

    passed = sum(results)
    print(f"\n=== {passed} OK / {len(results) - passed} FAIL ===")

    if not results[2]:
        print(
            "\nツール呼び出しが通りませんでした。このモデルでは 17 個のツールを"
            "扱いきれない可能性があります。別のモデルを試すか、"
            "TOKA_EMOTION_MODE=separate で感情更新を function calling から"
            "切り離してください。"
        )
    return 0 if passed == len(results) else 1


async def run_chat(provider: str, model: str | None) -> int:
    """対話して感触を見る。音声を通さずに応答の質だけを確かめられる。"""
    try:
        backend = factory.build(provider, model=model)
    except BackendError as exc:
        print(f"バックエンドを作れません: {exc}")
        return 1

    print(f"=== {backend.name} との対話（空行か Ctrl+C で終了）===\n")
    history: list[Message] = []

    try:
        while True:
            try:
                user = input("あなた> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user:
                break

            history.append(Message(role="user", text=user))
            print("トーカ> ", end="", flush=True)

            reply = ""
            started = time.monotonic()
            try:
                async for delta in backend.stream(
                    history,
                    system="あなたは「トーカ」。日本語で簡潔に、淡々と話します。",
                    temperature=0.8,
                ):
                    if delta.text:
                        print(delta.text, end="", flush=True)
                        reply += delta.text
            except BackendError as exc:
                print(f"\n(エラー: {exc})")
                history.pop()
                continue

            print(f"\n  [{time.monotonic() - started:.1f}秒]\n")
            history.append(Message(role="assistant", text=reply))
    finally:
        await backend.close()

    return 0


async def run_list_models(provider: str) -> int:
    models = await factory.list_models(provider)
    if not models:
        print("モデル一覧を取得できませんでした。")
        return 1
    print(f"=== {provider} で使えるモデル ({len(models)} 件) ===")
    for name in models:
        print(f"  {name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="python -m toka.llm",
        description="LLM バックエンドの診断",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="プリセット:\n" + factory.describe(),
    )
    parser.add_argument("--provider", help=f"({', '.join(factory.available())})")
    parser.add_argument("--model", help="既定モデルを上書きする")
    parser.add_argument("--smoke", action="store_true", help="4 項目を叩く")
    parser.add_argument("--chat", action="store_true", help="対話する")
    parser.add_argument("--list", action="store_true", help="プリセット一覧")
    parser.add_argument("--list-models", action="store_true", help="モデル一覧")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    load_dotenv()

    if args.list or not args.provider:
        print("プリセット:")
        print(factory.describe())
        if not args.provider:
            print("\n--provider <name> を指定してください。")
        return 0

    if args.list_models:
        return asyncio.run(run_list_models(args.provider))
    if args.chat:
        return asyncio.run(run_chat(args.provider, args.model))
    return asyncio.run(run_smoke(args.provider, args.model))


if __name__ == "__main__":
    raise SystemExit(main())
