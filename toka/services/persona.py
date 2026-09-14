"""話し方のモード。会話モードとアシスタントモードを声で切り替える。

きっかけは「検索できないのに調べたふりをする」という実際の不満だった。
ローカルの小さいモデルは、知らないことを知っているふうに埋めてくる。
かといって常に厳密だと、雑談相手としては固すぎる。

そこで 2 段にした。

    会話モード（既定）    冗談を言ってよい。多少あやふやでも話を回す
    アシスタントモード    事実の正確さを最優先。知らないなら知らないと言う

**ただし「使っていないツールを使ったふりをする」ことは、どちらのモードでも
禁止する。** これはモードで緩めていい部分ではない。正直さのダイヤルは
「確信のないことをどう言うか」の話で、「やっていないことをやったと言う」
のは別種の嘘である。前者は会話の潤滑油になりうるが、後者はアシスタント
としての信頼を壊す。

切り替えは決定論的なキーワード判定で行い、LLM のツール呼び出しには
頼らない。ローカルモデルはツールを呼ばないことがあり（実測で remember が
呼ばれなかった）、モードの切り替えまで取りこぼすと直しようがない。
"""

from __future__ import annotations

import logging
from enum import Enum

from .. import config

log = logging.getLogger(__name__)


class Mode(str, Enum):
    CONVERSATION = "conversation"
    ASSISTANT = "assistant"


# 「モード」と明示しているものは、文中のどこにあっても切り替えとみなす。
STRONG_ASSISTANT = (
    "アシスタントモード", "まじめモード", "真面目モード", "正確モード",
)
STRONG_CONVERSATION = (
    "会話モード", "雑談モード", "普通モード", "通常モード",
)

# こちらは普通の会話にも出てくる言い方なので、発話が短いときだけ拾う。
# 「まじめに働いてるんだけどさ、今日は……」で切り替わっては困る。
WEAK_ASSISTANT = (
    "まじめに", "真面目に", "正確に", "真剣に", "厳密に",
)
WEAK_CONVERSATION = (
    "普通に戻", "いつも通り", "いつもどおり", "気楽に", "砕けて",
)

# 弱いキーワードを切り替えとみなす発話の長さの上限（文字）。
# 「まじめに答えて」は 7 文字。これを超える発話は、命令ではなく
# 話の一部として「まじめに」を使っていると判断する。
WEAK_MAX_CHARS = 12


def detect(text: str) -> Mode | None:
    """発話からモードの切り替え指示を読み取る。無ければ None。"""
    if not text:
        return None
    stripped = text.strip()

    for word in STRONG_ASSISTANT:
        if word in stripped:
            return Mode.ASSISTANT
    for word in STRONG_CONVERSATION:
        if word in stripped:
            return Mode.CONVERSATION

    if len(stripped) <= WEAK_MAX_CHARS:
        for word in WEAK_ASSISTANT:
            if word in stripped:
                return Mode.ASSISTANT
        for word in WEAK_CONVERSATION:
            if word in stripped:
                return Mode.CONVERSATION

    return None


def strip_triggers(text: str) -> str:
    """切り替えの合図を発話から落とす。

    「まじめに、さっきの件どうなった？」と言われたとき、モードを変えた上で
    「さっきの件どうなった？」を質問として扱いたい。合図だけの発話なら
    空文字になる（呼び出し側で「了解」と返す）。
    """
    cleaned = (text or "").strip()
    for word in (*STRONG_ASSISTANT, *STRONG_CONVERSATION,
                 *WEAK_ASSISTANT, *WEAK_CONVERSATION):
        cleaned = cleaned.replace(word, "")
    # 合図を抜いた跡に残る助詞や記号を落とす。
    return cleaned.strip("　 、。,.！!？?にでへをはがのねよ")


def dials(mode: Mode) -> tuple[int, int]:
    """(正直度, ユーモア) を返す。0〜100。"""
    if mode is Mode.ASSISTANT:
        return config.ASSISTANT_HONESTY, config.ASSISTANT_HUMOR
    return config.CONVERSATION_HONESTY, config.CONVERSATION_HUMOR


def instructions(mode: Mode) -> str:
    """システムプロンプトへ差し込むモードの指示。

    数値だけ渡しても振る舞いは変わらないので、その値が意味する具体的な
    振る舞いを併記する。数値は config で調整できる。
    """
    honesty, humor = dials(mode)

    if mode is Mode.ASSISTANT:
        return (
            f"\n【今のモード】アシスタントモード（正直度 {honesty} / ユーモア {humor}）\n"
            "- 事実の正確さを最優先する。冗談や軽口は言わない。\n"
            "- 知らないことは「分からない」と言う。推測で埋めない。\n"
            "- 確認できていないことは、確認できていないと明示する。\n"
            "- 調べられることはツールで調べる。ツールが使えないなら"
            "「使えない」と言う。\n"
            "- ユーザーが「会話モード」と言ったら普段の話し方に戻る。"
        )

    return (
        f"\n【今のモード】会話モード（正直度 {honesty} / ユーモア {humor}）\n"
        "- 雑談として気楽に話す。冗談や軽口を混ぜてよい。\n"
        f"- ユーモアは {humor} 程度。面白がらせに行きすぎず、"
        "会話の流れで自然に出す程度にする。\n"
        f"- 正直度は {honesty}。事実として言い切るのは自分が知っていることだけに"
        "する。あやふやなことは「たぶん」「うろ覚えだけど」と添えて話す。\n"
        "- ただし冗談と嘘は別。ツールが失敗したら失敗と言う。"
        "調べていないことを調べたように話すのは、このモードでも禁止。\n"
        "- ユーザーが「まじめに」「アシスタントモード」と言ったら"
        "正確さ優先へ切り替わる。"
    )
