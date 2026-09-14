"""LLMService の履歴管理と文分割。

履歴の整合はプロバイダを替えて初めて問題になった部分である。OpenAI 互換は
「tool_calls を持つ assistant の直後に対応する tool が並んでいない」履歴を
400 で弾くため、割り込みやトリムで対を壊さないことを保証する必要がある。
Gemini はそこが緩かったので、以前は壊れていても気づけなかった。
"""

from __future__ import annotations

from toka import config
from toka.llm.types import Message, ToolCall
from toka.services.llm import SentenceSplitter


class TestSentenceSplitter:
    def test_句点で切る(self):
        s = SentenceSplitter()
        # ？ も文末なので「元気？」も完成した文として切り出される。
        assert s.feed("こんにちは。元気？") == ["こんにちは。", "元気？"]
        assert s.flush() == ""

    def test_文末が来ていない分は残す(self):
        s = SentenceSplitter()
        assert s.feed("こんにちは。まだ") == ["こんにちは。"]
        assert s.flush() == "まだ"

    def test_複数の文を一度に切る(self):
        s = SentenceSplitter()
        assert s.feed("一つ。二つ！三つ？") == ["一つ。", "二つ！", "三つ？"]

    def test_装飾記号を落とす(self):
        """読み上げるので Markdown は音にならない。"""
        s = SentenceSplitter()
        assert s.feed("**強調**です。") == ["強調です。"]

    def test_URL_を落とす(self):
        s = SentenceSplitter()
        assert s.feed("詳細は[ここ](https://example.com)です。") == ["詳細はここです。"]

    def test_句点が来なければ溜める(self):
        s = SentenceSplitter()
        assert s.feed("まだ途中") == []
        assert s.flush() == "まだ途中"

    def test_長すぎるものは読点で妥協して切る(self):
        """切らないと最初の音が出るまで待たされ続ける。"""
        s = SentenceSplitter()
        text = "あ" * 40 + "、" + "い" * 30
        out = s.feed(text)
        assert out
        assert out[0].endswith("、")

    def test_読点も無ければ文字数で切る(self):
        s = SentenceSplitter()
        out = s.feed("あ" * 200)
        assert out
        assert all(len(x) <= 60 for x in out)

    def test_空白だけの文は捨てる(self):
        s = SentenceSplitter()
        assert s.feed("   \n  ") == []

    def test_flush_は状態を空にする(self):
        s = SentenceSplitter()
        s.feed("途中")
        s.flush()
        assert s.flush() == ""


class _FakeMemory:
    def profile_text(self):
        return ""


def _service():
    """外部依存を持たない LLMService を組む。履歴操作だけを見る。"""
    from toka.services.llm import LLMService

    class _FakeRegistry:
        def declarations(self):
            return []

        def get(self, name):
            return None

    return LLMService(
        bus=None, router=None, memory=_FakeMemory(),
        registry=_FakeRegistry(), tts=None,
    )


class TestHistoryIntegrity:
    def test_結果の無いツール呼び出しを補完する(self):
        """割り込みでツール実行中に切られると、対が壊れて次のターンが 400 になる。"""
        service = _service()
        call = ToolCall(id="c1", name="open_app", args={"name": "メモ帳"})
        service._history = [
            Message(role="user", text="メモ帳を開いて"),
            Message(role="assistant", tool_calls=[call]),
        ]

        service._heal_dangling_tool_calls()

        assert service._history[-1].role == "tool"
        assert service._history[-1].tool_call_id == "c1"

    def test_結果が揃っていれば何も足さない(self):
        service = _service()
        call = ToolCall(id="c1", name="open_app", args={})
        service._history = [
            Message(role="assistant", tool_calls=[call]),
            Message(role="tool", tool_call_id="c1", text="開きました"),
        ]
        before = len(service._history)
        service._heal_dangling_tool_calls()
        assert len(service._history) == before

    def test_複数の呼び出しのうち欠けた分だけ補完する(self):
        service = _service()
        calls = [
            ToolCall(id="c1", name="a", args={}),
            ToolCall(id="c2", name="b", args={}),
        ]
        service._history = [
            Message(role="assistant", tool_calls=calls),
            Message(role="tool", tool_call_id="c1", text="ok"),
        ]
        service._heal_dangling_tool_calls()
        ids = [m.tool_call_id for m in service._history if m.role == "tool"]
        assert sorted(ids) == ["c1", "c2"]

    def test_トリムで孤立したツール結果を先頭に残さない(self):
        """先頭が role=tool の履歴は、対応する呼び出しを失っていて 400 になる。"""
        service = _service()
        limit = config.MAX_HISTORY_TURNS * 2
        # 先頭が tool になるように埋める。
        service._history = (
            [Message(role="user", text=f"u{i}") for i in range(limit)]
            + [Message(role="tool", tool_call_id="c1", text="r")]
            + [Message(role="user", text="最後")]
        )
        service._trim_history()
        assert service._history[0].role != "tool"

    def test_上限以下なら触らない(self):
        service = _service()
        service._history = [Message(role="user", text="a")]
        service._trim_history()
        assert len(service._history) == 1

    def test_割り込みの目印は_assistant_が続くときは足さない(self):
        service = _service()
        service._history = [Message(role="assistant", text="言いかけ")]
        service.note_interruption()
        assert len(service._history) == 1

    def test_割り込みの目印を足す(self):
        service = _service()
        service._history = [Message(role="user", text="やあ")]
        service.note_interruption()
        assert service._history[-1].role == "assistant"
        assert "遮られた" in service._history[-1].text


class TestSpeakableOutput:
    """読み上げに乗せてはいけない記号を落とす。

    小さいモデルは JSON の断片を応答へ混ぜてくる。実測では会話モードで
    `{"searchresults": }` がそのまま応答になった。記号を音読させない。
    """

    def test_波括弧を落とす(self):
        s = SentenceSplitter()
        assert s.feed('{"searchresults": }です。') == ["searchresults: です。"]

    def test_二重引用符を落とす(self):
        s = SentenceSplitter()
        assert s.feed('"晴れ"です。') == ["晴れです。"]

    def test_日本語の鉤括弧は残す(self):
        """「」は読み上げの区切りとして自然なので落とさない。"""
        s = SentenceSplitter()
        assert s.feed("「晴れ」です。") == ["「晴れ」です。"]

    def test_普通の文は変わらない(self):
        s = SentenceSplitter()
        assert s.feed("今日は晴れです。") == ["今日は晴れです。"]
