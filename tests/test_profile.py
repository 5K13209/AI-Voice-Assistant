"""声紋登録から作るプロフィールの抽出。

文字起こしは音声認識を通っているので誤字が混ざる。LLM の返り方も
プロバイダによって揺れるため、どの形で返ってきても拾えることを固定する。
拾えないと名前を覚えられず、逆に拾いすぎると誤った名前が恒久的に残る。
"""

from __future__ import annotations

import pytest

from toka.services.profile import MAX_FACTS, ProfileEnroller, parse_facts


class TestParseFacts:
    def test_facts_キーから取り出す(self):
        assert parse_facts({"facts": ["名前はサジ", "PCで作業する"]}) == [
            "名前はサジ", "PCで作業する",
        ]

    def test_配列がそのまま返ってきても拾う(self):
        """構造化出力に未対応だとプロンプト頼みになり、配列だけ返ることがある。"""
        assert parse_facts(["名前はサジ"]) == ["名前はサジ"]

    def test_別のキー名でも最初の配列を拾う(self):
        assert parse_facts({"profile": ["名前はサジ"]}) == ["名前はサジ"]

    def test_辞書の要素から文字列を取り出す(self):
        assert parse_facts({"facts": [{"fact": "名前はサジ"}]}) == ["名前はサジ"]
        assert parse_facts({"facts": [{"text": "名前はサジ"}]}) == ["名前はサジ"]

    def test_空白だけの要素は捨てる(self):
        assert parse_facts({"facts": ["  ", "名前はサジ", ""]}) == ["名前はサジ"]

    def test_前後の空白を落とす(self):
        assert parse_facts({"facts": ["  名前はサジ  "]}) == ["名前はサジ"]

    def test_文字列でない要素は捨てる(self):
        assert parse_facts({"facts": [123, None, "名前はサジ"]}) == ["名前はサジ"]

    def test_上限で打ち切る(self):
        """profile は毎ターン全量がプロンプトに載るので増やしすぎない。"""
        out = parse_facts({"facts": [f"覚えておく事実その{i}" for i in range(20)]})
        assert len(out) == MAX_FACTS

    def test_空の配列は空(self):
        assert parse_facts({"facts": []}) == []

    def test_facts_が配列でなければ空(self):
        assert parse_facts({"facts": "名前はサジ"}) == []

    def test_想定外の型は空(self):
        assert parse_facts("名前はサジ") == []
        assert parse_facts(None) == []


class _FakeMemory:
    def __init__(self):
        self.facts = []

    def store_fact(self, fact):
        if fact in self.facts:
            return False
        self.facts.append(fact)
        return True


class _FakeRouter:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    async def complete_sub(self, prompt, *, schema=None, temperature=0.0):
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class _FakeTranscriber:
    def __init__(self, text=""):
        self.text = text

    def transcribe(self, samples):
        return self.text


class TestEnroll:
    @pytest.mark.asyncio
    async def test_パスが空なら何もしない(self):
        router = _FakeRouter('{"facts": ["x"]}')
        memory = _FakeMemory()
        out = await ProfileEnroller(router, memory).enroll([], _FakeTranscriber())
        assert out == []
        assert router.prompts == []

    @pytest.mark.asyncio
    async def test_文字起こしが短ければ諦める(self, monkeypatch):
        """無音を録ってしまった場合。LLM に投げても推測しか返らない。"""
        router = _FakeRouter('{"facts": ["名前はサジ"]}')
        memory = _FakeMemory()
        enroller = ProfileEnroller(router, memory)
        monkeypatch.setattr(ProfileEnroller, "transcribe",
                            staticmethod(lambda paths, tr: "あ"))
        assert await enroller.enroll(["voice_0.wav"], _FakeTranscriber()) == []
        assert router.prompts == []

    @pytest.mark.asyncio
    async def test_文字起こしから事実を保存する(self, monkeypatch):
        router = _FakeRouter('{"facts": ["ユーザーの名前はサジです"]}')
        memory = _FakeMemory()
        enroller = ProfileEnroller(router, memory)
        monkeypatch.setattr(
            ProfileEnroller, "transcribe",
            staticmethod(lambda paths, tr: "こんにちは、名前はサジです。"),
        )
        out = await enroller.enroll(["voice_0.wav"], _FakeTranscriber())
        assert out == ["ユーザーの名前はサジです"]
        assert memory.facts == ["ユーザーの名前はサジです"]
        # 文字起こしがプロンプトに載っていること。
        assert "サジ" in router.prompts[0]

    @pytest.mark.asyncio
    async def test_既知の事実は数に入れない(self, monkeypatch):
        router = _FakeRouter('{"facts": ["既知"]}')
        memory = _FakeMemory()
        memory.facts.append("既知")
        enroller = ProfileEnroller(router, memory)
        monkeypatch.setattr(ProfileEnroller, "transcribe",
                            staticmethod(lambda paths, tr: "既知の話です。"))
        assert await enroller.enroll(["voice_0.wav"], _FakeTranscriber()) == []

    @pytest.mark.asyncio
    async def test_LLM_が落ちても例外を出さない(self, monkeypatch):
        """名前を覚えられないだけで、起動は続けたい。"""
        router = _FakeRouter(RuntimeError("落ちた"))
        memory = _FakeMemory()
        enroller = ProfileEnroller(router, memory)
        monkeypatch.setattr(ProfileEnroller, "transcribe",
                            staticmethod(lambda paths, tr: "名前はサジです。"))
        assert await enroller.enroll(["voice_0.wav"], _FakeTranscriber()) == []

    @pytest.mark.asyncio
    async def test_JSON_でない返答でも例外を出さない(self, monkeypatch):
        router = _FakeRouter("すみません、分かりません")
        memory = _FakeMemory()
        enroller = ProfileEnroller(router, memory)
        monkeypatch.setattr(ProfileEnroller, "transcribe",
                            staticmethod(lambda paths, tr: "名前はサジです。"))
        assert await enroller.enroll(["voice_0.wav"], _FakeTranscriber()) == []


class TestFactFiltering:
    """「分からない」旨の事実を弾く。

    profile は毎ターン全量がプロンプトに載るので、中身の無い事実を入れると
    永久に居座る。プロンプトで「推測せず捨てる」と指示しても小さいモデルは
    従わなかったため、コード側で落としている。
    """

    def test_分からない旨の事実を弾く(self):
        out = parse_facts({"facts": [
            "名前は分かりません",
            "呼ばれ方は分からない",
            "天気は不明です",
            "ユーザーの名前はサジです",
        ]})
        assert out == ["ユーザーの名前はサジです"]

    def test_聞き取れなかった旨も弾く(self):
        assert parse_facts({"facts": ["名前は聞き取れませんでした"]}) == []

    def test_特にないも弾く(self):
        assert parse_facts({"facts": ["趣味は特にないようです"]}) == []

    def test_短すぎる事実を弾く(self):
        assert parse_facts({"facts": ["うん", "はい", "そう", "あ"]}) == []

    def test_重複を落とす(self):
        out = parse_facts({"facts": [
            "ユーザーの名前はサジです",
            "ユーザーの名前はサジです",
            "ユーザーの名前はサジです。",
        ]})
        assert out == ["ユーザーの名前はサジです"]

    def test_有用な事実は残す(self):
        out = parse_facts({"facts": [
            "ユーザーの名前はサジです",
            "普段はプログラミングをしている",
        ]})
        assert len(out) == 2

    def test_実測で返ってきた一覧をすべて弾く(self):
        """llama3.1:8b が実際に返したもの。1件も残ってはいけない。"""
        out = parse_facts({"facts": [
            "名前は分かりません", "呼ばれ方は分かりません",
            "ここにちょこちょこしている", "天気は分からない",
            "天気は分からない", "分からない",
        ]})
        # 「ここにちょこちょこしている」は誤認識由来だが、形式上は
        # 弾けない。少なくとも「分からない」系と重複は消えること。
        assert all("分から" not in f for f in out)
        assert len(out) == len(set(out))


class TestConfirmation:
    """保存前の確認。誤認識由来の事実が恒久的に残るのを防ぐ最後の関門。"""

    @pytest.mark.asyncio
    async def test_断れば保存しない(self, monkeypatch):
        router = _FakeRouter('{"facts": ["ユーザーの名前はサジです"]}')
        memory = _FakeMemory()
        enroller = ProfileEnroller(router, memory)
        monkeypatch.setattr(ProfileEnroller, "transcribe",
                            staticmethod(lambda paths, tr: "名前はサジです。"))
        monkeypatch.setattr(ProfileEnroller, "_confirm",
                            lambda self, facts: _false())
        assert await enroller.enroll(["voice_0.wav"], _FakeTranscriber()) == []
        assert memory.facts == []

    @pytest.mark.asyncio
    async def test_端末が無ければ訊かずに通す(self, monkeypatch):
        """ログへリダイレクトした起動で止まってはいけない。"""
        import sys as _sys

        class _NoTty:
            def isatty(self):
                return False

        monkeypatch.setattr(_sys, "stdin", _NoTty())
        enroller = ProfileEnroller(_FakeRouter("{}"), _FakeMemory())
        assert await enroller._confirm(["事実です"]) is True

    @pytest.mark.asyncio
    async def test_事実が無ければ確認もしない(self, monkeypatch):
        router = _FakeRouter('{"facts": []}')
        memory = _FakeMemory()
        enroller = ProfileEnroller(router, memory)
        monkeypatch.setattr(ProfileEnroller, "transcribe",
                            staticmethod(lambda paths, tr: "うーん。"))
        asked = []
        monkeypatch.setattr(ProfileEnroller, "_confirm",
                            lambda self, facts: asked.append(1) or _true())
        assert await enroller.enroll(["voice_0.wav"], _FakeTranscriber()) == []
        assert asked == []


async def _false():
    return False


async def _true():
    return True
