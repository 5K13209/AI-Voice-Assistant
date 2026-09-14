"""声紋登録の音声から、ユーザーのプロフィールを取り出す。

声紋を取るには、どうせ十数秒喋ってもらう必要がある。その音声を捨てずに
文字起こしして、名前や普段していることを長期記憶（profile 層）へ入れる。
登録が 1 回で済み、初回の会話からユーザーの名前を呼べるようになる。

文字起こしは音声認識を通っているので誤字が混ざる。そのまま事実として
保存すると「ユーザーの名前はサジです」のような取り違えが恒久的に残るため、
LLM に整えさせてから入れる。確信が持てない項目は捨てさせる。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

from .. import config

log = logging.getLogger(__name__)

# 1 回の登録で覚える事実の上限。profile は毎ターン全量がプロンプトに載るので、
# 増やしすぎるとコンテキストを食い続ける。
MAX_FACTS = 6

# 文字起こしがこれより短ければ、何も喋っていないものとして諦める。
MIN_TRANSCRIPT_CHARS = 4

# 事実として短すぎるものは中身が無い。「うん」「はい」など相槌を落とす。
# 「分からない」系は文字数ではなく IGNORANCE_MARKERS で落とすので、ここは
# 低めにしておく。高くすると「名前はサジ」のような正当な事実まで消える。
MIN_FACT_CHARS = 4

# 「分からない」旨の事実を弾く。プロンプトで「推測せず捨てる」と指示しても、
# 小さいモデルは素直に「名前は分かりません」を事実として返してくる。実測で
# llama3.1:8b が返したのが次のようなものだった:
#   ['名前は分かりません', '呼ばれ方は分かりません', '天気は分からない', '分からない']
# これを profile に入れると毎ターンのプロンプトに永久に載り、覚えないより
# 悪い。指示ではなくコード側で落とす。
#
# 「分からないことは訊いてほしい」のような正当な事実も巻き込みうるが、
# 誤った断定を恒久的に持つ方が害が大きいので、こちらへ倒す。
# 活用形を落とさないこと。「分から」だけでは「分かりません」に一致しない
# （分・か・り なので）。実測で最初に漏れたのがまさにこれだった。
IGNORANCE_MARKERS = (
    "分から", "分かりま", "分かんな", "分りま",
    "わから", "わかりま", "わかんな",
    "不明", "不詳", "判別できな",
    "特にな", "該当な", "情報がな", "言及がな", "記載がな",
    "聞き取れ", "認識できな", "読み取れな",
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "description": "ユーザーについて分かった事実。日本語の短い平叙文",
            "items": {"type": "string"},
        }
    },
    "required": ["facts"],
}

_PROMPT = """次の文章は、ユーザーが自己紹介として話した音声を文字起こししたものです。

音声認識を通しているため、誤認識や言い間違いが混ざっています。
ここからユーザーについての事実を取り出してください。

規則:
- 名前や呼ばれ方が分かるなら、それを最優先で入れてください。
- 「ユーザーの名前は○○です」「○○と呼ばれたい」のような短い平叙文にする。
- 長く覚えておく価値のあることだけ。その場限りの話題は入れない。
- 聞き取れていない、意味が取れない箇所は**推測せずに捨てる**。
  誤った名前を覚える方が、覚えないより悪い結果になります。
- 何も取り出せなければ空の配列を返してください。
- 最大{max_facts}件。

【文字起こし】
{transcript}
"""


def strip_stop_words(text: str) -> str:
    """文字起こしの末尾から録音終了の合図を落とす。

    録音は「以上」と言うと止まる仕組みなので、その語が必ず末尾に残る。
    落とさないと「以上と言いました」のような事実を作りかねない。
    末尾だけを見る（文中の「以上」は本文なので残す）。
    """
    from .auth import PUNCTUATION, peel_polite

    cleaned = (text or "").strip()
    for _ in range(3):  # 「終わり。以上です」のように重なることがある
        stripped = peel_polite(cleaned)
        for word in config.VOICE_ENROLL_STOP_WORDS:
            if stripped.endswith(word):
                stripped = stripped[: -len(word)]
                break
        else:
            break
        cleaned = stripped
    return cleaned.strip(PUNCTUATION)


class ProfileEnroller:
    """登録音声 -> 文字起こし -> 事実の抽出 -> 長期記憶。"""

    def __init__(self, router, memory) -> None:
        self.router = router
        self.memory = memory

    @staticmethod
    def transcribe(paths: list[str], transcriber) -> str:
        """登録音声をまとめて文字起こしする。CPU バウンドなので executor で。"""
        import soundfile as sf

        from .auth import resolve_ref

        chunks = []
        for path in paths:
            try:
                samples, rate = sf.read(
                    resolve_ref(path), dtype="float32", always_2d=False
                )
            except Exception:
                log.warning("登録音声を読めません: %s", path)
                continue

            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            if rate != config.SAMPLE_RATE:
                # 認識器は 16kHz 固定。変換せずに渡すと中身が別物になる。
                from math import gcd

                from scipy.signal import resample_poly

                divisor = gcd(config.SAMPLE_RATE, rate)
                samples = resample_poly(
                    samples, config.SAMPLE_RATE // divisor, rate // divisor
                )

            text = strip_stop_words(transcriber.transcribe(samples))
            if text:
                chunks.append(text)

        return "\n".join(chunks)

    async def enroll(self, paths: list[str], transcriber) -> list[str]:
        """登録音声からプロフィールを作る。保存できた事実を返す。

        失敗しても起動は止めない。名前を覚えられなかっただけで、
        会話そのものは成立するため。
        """
        if not paths:
            return []

        loop = asyncio.get_running_loop()
        transcript = await loop.run_in_executor(
            None, self.transcribe, paths, transcriber
        )
        transcript = transcript.strip()

        if len(transcript) < MIN_TRANSCRIPT_CHARS:
            log.info("登録音声から文字を取れなかったため、プロフィールは作りません")
            return []

        log.info("登録音声の文字起こし: %s", transcript.replace("\n", " / "))

        facts = await self._extract(transcript)
        if not facts:
            log.info("プロフィールとして残せる内容はありませんでした")
            return []

        # 保存前に必ず見せて確認を取る。profile は毎ターンのプロンプトに
        # 載り続けるので、誤認識由来の事実が一度入ると会話に効き続ける。
        # 実測でも、はっきり喋っていない音声から
        # 「ここにちょこちょこしている」のような事実が生成された。
        if not await self._confirm(facts):
            log.info("プロフィールの登録は見送りました")
            return []

        stored = [fact for fact in facts if self.memory.store_fact(fact)]
        if stored:
            print(f"\n{len(stored)} 件を覚えました。\n")
        return stored

    async def _confirm(self, facts: list[str]) -> bool:
        """覚える内容をユーザーに見せて可否を訊く。

        端末が無い場合（ログへのリダイレクト、テスト）は訊けないので、
        訊かずに通す。ここで止めると起動そのものが進まなくなる。
        """
        print("\n次の内容を覚えようとしています:")
        for fact in facts:
            print(f"  - {fact}")

        if not sys.stdin or not sys.stdin.isatty():
            print("（確認を省略して登録します）\n")
            return True

        loop = asyncio.get_running_loop()
        try:
            answer = await loop.run_in_executor(
                None, input, "これで覚えていい？ [Y/n]: "
            )
        except (EOFError, OSError):
            return True

        return answer.strip().lower() not in ("n", "no", "いいえ")

    async def _extract(self, transcript: str) -> list[str]:
        prompt = _PROMPT.format(max_facts=MAX_FACTS, transcript=transcript)
        try:
            raw = await self.router.complete_sub(
                prompt, schema=_SCHEMA, temperature=0.0
            )
        except Exception as exc:
            log.warning("プロフィールの抽出に失敗: %s", exc)
            return []

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            log.warning("プロフィールの抽出結果を解釈できません: %r", raw[:200])
            return []

        return parse_facts(parsed)


def parse_facts(parsed: object) -> list[str]:
    """抽出結果から事実の一覧を取り出す。

    構造化出力に未対応のプロバイダではプロンプトだけで JSON を頼むことに
    なるため、配列がそのまま返ることも、別のキーに入ることもある。
    どの形で来ても拾えるようにしておく。
    """
    if isinstance(parsed, list):
        candidates = parsed
    elif isinstance(parsed, dict):
        candidates = parsed.get("facts")
        if not isinstance(candidates, list):
            # facts 以外のキー名で返してくることがある。最初の配列を採る。
            candidates = next(
                (v for v in parsed.values() if isinstance(v, list)), []
            )
    else:
        return []

    facts: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if isinstance(item, dict):
            # {"fact": "..."} の形で返す実装がある。
            item = item.get("fact") or item.get("text") or ""
        if not isinstance(item, str):
            continue

        text = item.strip()
        if not is_useful_fact(text):
            continue

        # 同じ事実を何度も返してくることがある。重複は落とす。
        key = text.rstrip("。.")
        if key in seen:
            continue
        seen.add(key)
        facts.append(text)

    return facts[:MAX_FACTS]


def is_useful_fact(text: str) -> bool:
    """覚える価値のある事実かどうか。

    「分からない」旨のものと、短すぎて中身の無いものを落とす。
    """
    if len(text) < MIN_FACT_CHARS:
        return False
    return not any(marker in text for marker in IGNORANCE_MARKERS)
