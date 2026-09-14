"""ツール宣言の生成と感情計算。

いずれも純関数で外部依存が無い。プロバイダ差し替えでスキーマの型名を
小文字へ変えたので、その回帰をここで押さえる。
"""

from __future__ import annotations

from toka.services.emotion import KEYS, KeywordFallback, apply, clamp_delta
from toka.tools.registry import Tool


def _tool(func, **kwargs):
    return Tool(name=func.__name__, description="d", func=func, **kwargs)


class TestToolDeclaration:
    def test_型名は小文字の_JSON_Schema(self):
        """大文字は Gemini 固有。ここで吐くとプロバイダを替えられない。"""
        def f(name: str, count: int, ratio: float, flag: bool) -> str:
            return ""

        schema = _tool(f).declaration()["parameters"]
        assert schema["type"] == "object"
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["ratio"]["type"] == "number"
        assert schema["properties"]["flag"]["type"] == "boolean"

    def test_既定値の無い引数だけ_required(self):
        def f(a: str, b: str = "x") -> str:
            return ""

        assert _tool(f).declaration()["parameters"]["required"] == ["a"]

    def test_引数が無ければ_required_を入れない(self):
        def f() -> str:
            return ""

        assert "required" not in _tool(f).declaration()["parameters"]

    def test_Union_は最初の実型を採る(self):
        def f(a: str | None = None) -> str:
            return ""

        schema = _tool(f).declaration()["parameters"]
        assert schema["properties"]["a"]["type"] == "string"

    def test_未知の型は_string_に落とす(self):
        def f(a) -> str:
            return ""

        schema = _tool(f).declaration()["parameters"]
        assert schema["properties"]["a"]["type"] == "string"

    def test_params_の説明が入る(self):
        def f(city: str) -> str:
            return ""

        schema = _tool(f, params={"city": "都市名"}).declaration()["parameters"]
        assert schema["properties"]["city"]["description"] == "都市名"

    def test_説明が無ければ引数名を使う(self):
        def f(city: str) -> str:
            return ""

        schema = _tool(f).declaration()["parameters"]
        assert schema["properties"]["city"]["description"] == "city"

    def test_確認文に引数が入る(self):
        def f(name: str) -> str:
            return ""

        tool = _tool(f, risk="confirm", confirm_template="{args} を開いていい？")
        assert tool.question({"name": "メモ帳"}) == "name=メモ帳 を開いていい？"

    def test_引数なしの確認文(self):
        def f() -> str:
            return ""

        tool = _tool(f, risk="confirm", confirm_template="{args} を実行？")
        assert "引数なし" in tool.question({})


class TestClampDelta:
    def test_上限で頭打ちにする(self):
        """LLM が極端な値を返しても暴れないようにする。"""
        out = clamp_delta({k: 100 for k in KEYS})
        assert all(v == 5 for v in out.values())

    def test_下限でも頭打ちにする(self):
        out = clamp_delta({k: -100 for k in KEYS})
        assert all(v == -5 for v in out.values())

    def test_欠けたキーは_0(self):
        out = clamp_delta({"like": 3})
        assert out["like"] == 3
        assert out["fun"] == 0

    def test_数値でない値は_0(self):
        out = clamp_delta({"like": "たくさん"})
        assert out["like"] == 0

    def test_None_は_0(self):
        assert clamp_delta({"like": None})["like"] == 0

    def test_全キーが揃う(self):
        assert set(clamp_delta({}).keys()) == set(KEYS)


class TestApply:
    def test_差分を足す(self):
        out = apply({k: 50 for k in KEYS}, {"like": 5})
        assert out["like"] == 55

    def test_0_と_100_で止める(self):
        assert apply({k: 98 for k in KEYS}, {"like": 5})["like"] == 100
        assert apply({k: 2 for k in KEYS}, {"like": -5})["like"] == 0

    def test_元の_dict_を壊さない(self):
        """参照のまま持つと履歴が最新の感情値で塗り潰される。"""
        original = {k: 50 for k in KEYS}
        apply(original, {"like": 5})
        assert original["like"] == 50


class TestKeywordFallback:
    def test_好意的な語で上がる(self):
        out = KeywordFallback.delta("ありがとう")
        assert out["like"] > 0
        assert out["trust"] > 0

    def test_否定的な語で怒りが上がり信頼が下がる(self):
        out = KeywordFallback.delta("うるさい")
        assert out["anger"] > 0
        assert out["trust"] < 0

    def test_無関係な語では動かない(self):
        assert not any(KeywordFallback.delta("今日は水曜日").values())

    def test_全キーが揃う(self):
        assert set(KeywordFallback.delta("").keys()) == set(KEYS)
