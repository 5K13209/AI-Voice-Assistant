"""話し方のモードと、その切り替え判定。

きっかけは「検索できないのに調べたふりをする」という実際の不満。
切り替えは決定論的なキーワード判定で行うので、誤検出すると
普通の会話中に勝手にモードが変わる。そこを固定する。
"""

from __future__ import annotations

from toka import config
from toka.services.persona import (
    Mode,
    detect,
    dials,
    instructions,
    strip_triggers,
)


class TestDetect:
    def test_アシスタントモードで切り替わる(self):
        assert detect("アシスタントモードにして") is Mode.ASSISTANT

    def test_会話モードで戻る(self):
        assert detect("会話モードに戻して") is Mode.CONVERSATION

    def test_まじめにで切り替わる(self):
        assert detect("まじめに") is Mode.ASSISTANT
        assert detect("まじめに答えて") is Mode.ASSISTANT

    def test_正確にで切り替わる(self):
        assert detect("正確に教えて") is Mode.ASSISTANT

    def test_普通に戻ってで戻る(self):
        assert detect("普通に戻って") is Mode.CONVERSATION

    def test_長い発話の中のまじめには無視する(self):
        """「まじめに働いてる」で勝手に切り替わってはいけない。"""
        assert detect("まじめに働いてるんだけどさ、今日は疲れたよ") is None

    def test_長くてもモードと言えば切り替わる(self):
        """明示的な「モード」は文中どこでも拾う。"""
        text = "ちょっと調べてほしいから、アシスタントモードにしてくれる？"
        assert detect(text) is Mode.ASSISTANT

    def test_普通の発話では切り替わらない(self):
        assert detect("今日の天気を教えて") is None
        assert detect("音量を上げて") is None

    def test_空文字では切り替わらない(self):
        assert detect("") is None
        assert detect(None) is None

    def test_弱いキーワードの長さの境界(self):
        """境界を超えたら命令ではなく本文として扱う。"""
        from toka.services.persona import WEAK_MAX_CHARS

        just_in = "まじめに" + "あ" * (WEAK_MAX_CHARS - 4)
        assert len(just_in) == WEAK_MAX_CHARS
        assert detect(just_in) is Mode.ASSISTANT

        just_out = just_in + "あ"
        assert detect(just_out) is None


class TestStripTriggers:
    def test_合図だけなら空になる(self):
        assert strip_triggers("まじめに") == ""
        assert strip_triggers("アシスタントモード") == ""

    def test_残った質問を返す(self):
        assert strip_triggers("まじめに、さっきの件どうなった？") == "さっきの件どうなった"

    def test_モード語を抜いた残りを返す(self):
        out = strip_triggers("アシスタントモードで今日の天気を調べて")
        assert "今日の天気を調べて" in out

    def test_合図が無ければそのまま(self):
        assert strip_triggers("音量を上げて") == "音量を上げて"

    def test_空文字は空(self):
        assert strip_triggers("") == ""
        assert strip_triggers(None) == ""


class TestDials:
    def test_会話モードの値は設定から来る(self):
        honesty, humor = dials(Mode.CONVERSATION)
        assert honesty == config.CONVERSATION_HONESTY
        assert humor == config.CONVERSATION_HUMOR

    def test_アシスタントモードは正直度が最大(self):
        honesty, humor = dials(Mode.ASSISTANT)
        assert honesty == 100
        assert humor < config.CONVERSATION_HUMOR

    def test_会話モードは冗談を許す(self):
        """既定は正直度 90 / ユーモア 55。"""
        assert dials(Mode.CONVERSATION) == (90, 55)


class TestInstructions:
    def test_アシスタントモードは分からないと言わせる(self):
        text = instructions(Mode.ASSISTANT)
        assert "分からない" in text
        assert "冗談" in text

    def test_会話モードは冗談を許す(self):
        text = instructions(Mode.CONVERSATION)
        assert "冗談" in text

    def test_数値がプロンプトに載る(self):
        text = instructions(Mode.CONVERSATION)
        assert "90" in text
        assert "55" in text

    def test_戻り方を案内する(self):
        assert "会話モード" in instructions(Mode.ASSISTANT)
        assert "アシスタントモード" in instructions(Mode.CONVERSATION)

    def test_どちらのモードでも文字列を返す(self):
        for mode in Mode:
            assert instructions(mode).strip()


class TestAbsoluteRule:
    def test_ツールのふりを禁じる規則が常に載る(self):
        """モードで緩めてよい部分ではないので PERSONA 側に置いてある。"""
        from toka.services.llm import PERSONA

        assert "呼んでいないツールを呼んだことにしない" in PERSONA
        assert "例外なし" in PERSONA


class TestFailureNote:
    """失敗したツール結果に添える指示。

    「システムプロンプトに書いてあるから従うはず」が通らなかったので、
    モデルが次に読む場所（ツール結果そのもの）へ書いている。
    """

    def test_失敗した結果に指示が付く(self):
        from toka.services.llm import _failure_note

        note = _failure_note("TAVILY_API_KEY が設定されていません。")
        assert "実行失敗" in note
        assert "創作" in note
        # 元の失敗内容は消さない。ユーザーへ伝える必要がある。
        assert "TAVILY_API_KEY" in note

    def test_会話モードでも捏造を禁じる(self):
        """正直度 90 でも、ここは緩めてよい部分ではない。"""
        text = instructions(Mode.CONVERSATION)
        assert "禁止" in text
