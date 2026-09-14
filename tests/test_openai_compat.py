"""OpenAI 互換アダプタの単体テスト。

ここが今回の実装で最も壊れやすい。ストリーミングのツール呼び出しは
断片で届くため、組み立てを間違えるとツールが一切呼べなくなる。しかも
症状は「LLM がツールを使ってくれない」という形で出るので、原因が
プロンプトなのか実装なのか切り分けられない。フィクスチャで固定する。
"""

from __future__ import annotations

import json

import pytest

from toka.llm.openai_compat import (
    LeakedToolCallFilter,
    ThinkFilter,
    _ToolCallAccumulator,
    to_openai_messages,
    to_openai_tools,
)
from toka.llm.types import Message, ToolCall

KNOWN = frozenset({"get_volume", "open_app", "remember"})


def _run(filter_, chunks):
    """チャンク列を流し込んで、可視テキストを返す。"""
    out = "".join(filter_.feed(c) for c in chunks)
    return out + filter_.flush()


class TestLeakedToolCallFilter:
    """ローカルモデルで区切りトークンが化けたときの回収。

    実測された壊れ方をそのまま固定する。ここが効かないとツールが
    実行されず、しかも JSON が音声で読み上げられる。
    """

    def test_タイ語に化けた区切りごと回収する(self):
        f = LeakedToolCallFilter(KNOWN)
        leaked = 'คณะกรรม\n{"name": "get_volume", "arguments": {}}\nคณะกรรม'
        visible = _run(f, [leaked])
        assert f.salvaged[0].name == "get_volume"
        assert f.salvaged[0].args == {}
        # 化けた区切りも読み上げに出してはいけない。
        assert "คณะกรรม" not in visible
        assert "{" not in visible

    def test_icall_に化けた区切りでも回収する(self):
        f = LeakedToolCallFilter(KNOWN)
        leaked = '_icall_\n{"name": "get_volume", "arguments": {}}\n_ick_'
        visible = _run(f, [leaked])
        assert f.salvaged[0].name == "get_volume"
        assert "icall" not in visible

    def test_引数つきの呼び出しを回収する(self):
        f = LeakedToolCallFilter(KNOWN)
        _run(f, ['{"name": "open_app", "arguments": {"name": "メモ帳"}}'])
        assert f.salvaged[0].args == {"name": "メモ帳"}

    def test_チャンク境界で割れても回収する(self):
        f = LeakedToolCallFilter(KNOWN)
        visible = _run(f, ['_ica', 'll_\n{"name": "get_', 'volume", "argu',
                           'ments": {}}\n_ick_'])
        assert len(f.salvaged) == 1
        assert f.salvaged[0].name == "get_volume"
        assert "{" not in visible

    def test_引数が文字列で入っていても解く(self):
        f = LeakedToolCallFilter(KNOWN)
        _run(f, ['{"name": "open_app", "arguments": "{\\"name\\": \\"電卓\\"}"}'])
        assert f.salvaged[0].args == {"name": "電卓"}

    def test_未知の名前は回収しない(self):
        """ユーザーが読み上げてほしい JSON を勝手にツール呼び出しへ化かさない。"""
        f = LeakedToolCallFilter(KNOWN)
        visible = _run(f, ['{"name": "rm_rf", "arguments": {}}'])
        assert f.salvaged == []
        assert "rm_rf" in visible

    def test_ツール呼び出しでない_JSON_はそのまま流す(self):
        f = LeakedToolCallFilter(KNOWN)
        visible = _run(f, ['{"温度": 22, "天気": "晴れ"}'])
        assert f.salvaged == []
        assert "22" in visible

    def test_普通の応答は素通しする(self):
        f = LeakedToolCallFilter(KNOWN)
        assert _run(f, ["こんにちは。", "元気です。"]) == "こんにちは。元気です。"

    def test_長い応答でも先頭を落とさない(self):
        f = LeakedToolCallFilter(KNOWN)
        text = "あ" * 300
        assert _run(f, [text]) == text

    def test_判断がつく前に終わっても保留分を吐く(self):
        """短い応答は PROBE_CHARS に達しないまま終わる。捨ててはいけない。"""
        f = LeakedToolCallFilter(KNOWN)
        assert _run(f, ["はい。"]) == "はい。"

    def test_閉じない_JSON_は捨てる(self):
        f = LeakedToolCallFilter(KNOWN)
        visible = _run(f, ['{"name": "get_volume", "argum'])
        assert "{" not in visible

    def test_回収後の後続テキストは流す(self):
        f = LeakedToolCallFilter(KNOWN)
        visible = _run(f, ['{"name": "get_volume", "arguments": {}}',
                           "調べました。"])
        assert len(f.salvaged) == 1
        assert "調べました。" in visible

    def test_ツール名を渡さなければ形だけで判断する(self):
        f = LeakedToolCallFilter()
        _run(f, ['{"name": "whatever", "arguments": {}}'])
        assert f.salvaged[0].name == "whatever"

    def test_回収した呼び出しに_id_が付く(self):
        f = LeakedToolCallFilter(KNOWN)
        _run(f, ['{"name": "get_volume", "arguments": {}}'])
        assert f.salvaged[0].id


class TestToolCallAccumulator:
    def test_単一の呼び出しが断片で届く(self):
        """実際の OpenAI ストリームの形。name は最初だけ、args は細切れ。"""
        acc = _ToolCallAccumulator()
        acc.feed([{"index": 0, "id": "call_abc", "type": "function",
                   "function": {"name": "open_app", "arguments": ""}}])
        acc.feed([{"index": 0, "function": {"arguments": '{"na'}}])
        acc.feed([{"index": 0, "function": {"arguments": 'me": "メ'}}])
        acc.feed([{"index": 0, "function": {"arguments": 'モ帳"}'}}])

        calls = acc.finish()
        assert len(calls) == 1
        assert calls[0].id == "call_abc"
        assert calls[0].name == "open_app"
        assert calls[0].args == {"name": "メモ帳"}

    def test_複数の呼び出しが交互に届く(self):
        """並行するツール呼び出しは index で区別される。混ざってはいけない。"""
        acc = _ToolCallAccumulator()
        acc.feed([
            {"index": 0, "id": "a", "function": {"name": "set_volume", "arguments": ""}},
            {"index": 1, "id": "b", "function": {"name": "open_app", "arguments": ""}},
        ])
        acc.feed([{"index": 0, "function": {"arguments": '{"percent"'}}])
        acc.feed([{"index": 1, "function": {"arguments": '{"name":'}}])
        acc.feed([{"index": 0, "function": {"arguments": ": 30}"}}])
        acc.feed([{"index": 1, "function": {"arguments": ' "電卓"}'}}])

        calls = acc.finish()
        assert [c.name for c in calls] == ["set_volume", "open_app"]
        assert calls[0].args == {"percent": 30}
        assert calls[1].args == {"name": "電卓"}

    def test_index_が無い実装でも受け付ける(self):
        acc = _ToolCallAccumulator()
        acc.feed([{"id": "x", "function": {"name": "get_volume", "arguments": "{}"}}])
        calls = acc.finish()
        assert len(calls) == 1
        assert calls[0].name == "get_volume"
        assert calls[0].args == {}

    def test_引数が最初から_dict_で来る実装(self):
        """Ollama など、パース済みで返す実装がある。"""
        acc = _ToolCallAccumulator()
        acc.feed([{"index": 0, "id": "x",
                   "function": {"name": "open_app", "arguments": {"name": "電卓"}}}])
        calls = acc.finish()
        assert calls[0].args == {"name": "電卓"}

    def test_id_が無ければ採番する(self):
        """履歴でツール結果と突き合わせるため、id は必ず必要。"""
        acc = _ToolCallAccumulator()
        acc.feed([{"index": 0, "function": {"name": "get_volume", "arguments": "{}"}}])
        calls = acc.finish()
        assert calls[0].id
        assert "get_volume" in calls[0].id

    def test_壊れた引数は空_dict_にする(self):
        """例外を投げずに通す。llm.py が TypeError を LLM に返して直させる。"""
        acc = _ToolCallAccumulator()
        acc.feed([{"index": 0, "id": "x",
                   "function": {"name": "open_app", "arguments": '{"name": '}}])
        calls = acc.finish()
        assert len(calls) == 1
        assert calls[0].args == {}

    def test_引数が空文字なら引数なしとみなす(self):
        acc = _ToolCallAccumulator()
        acc.feed([{"index": 0, "id": "x",
                   "function": {"name": "get_volume", "arguments": ""}}])
        assert acc.finish()[0].args == {}

    def test_名前の無い断片は捨てる(self):
        acc = _ToolCallAccumulator()
        acc.feed([{"index": 0, "function": {"arguments": "{}"}}])
        assert acc.finish() == []

    def test_呼び出しが無ければ空(self):
        acc = _ToolCallAccumulator()
        assert acc.finish() == []
        assert not acc


class TestThinkFilter:
    def test_思考ブロックを落とす(self):
        f = ThinkFilter()
        assert f.feed("<think>ここは思考</think>こんにちは") == "こんにちは"

    def test_タグがチャンク境界で割れても落とす(self):
        """ストリーミングでは開始タグ自体が分割されて届く。"""
        f = ThinkFilter()
        out = "".join([
            f.feed("答えは"),
            f.feed("<thi"),
            f.feed("nk>内部の"),
            f.feed("思考</thi"),
            f.feed("nk>42です"),
        ])
        assert out + f.flush() == "答えは42です"

    def test_タグの部分一致を出力してしまわない(self):
        """"<thi" の時点で吐くと、後で本物のタグだと分かっても手遅れ。"""
        f = ThinkFilter()
        assert "<" not in f.feed("計算<thi")

    def test_タグに見えたが違った場合は出力する(self):
        f = ThinkFilter()
        emitted = f.feed("a<b") + f.flush()
        assert emitted == "a<b"

    def test_思考が閉じないまま終わったら捨てる(self):
        f = ThinkFilter()
        f.feed("<think>閉じない")
        assert f.flush() == ""

    def test_思考が無ければそのまま通す(self):
        f = ThinkFilter()
        assert f.feed("普通の応答") + f.flush() == "普通の応答"


class TestMessageConversion:
    def test_system_は先頭要素になる(self):
        """Gemini は config のフィールドだが、OpenAI 互換は messages[0]。"""
        out = to_openai_messages([Message(role="user", text="やあ")], "君はトーカ")
        assert out[0] == {"role": "system", "content": "君はトーカ"}
        assert out[1] == {"role": "user", "content": "やあ"}

    def test_system_が空なら入れない(self):
        out = to_openai_messages([Message(role="user", text="やあ")])
        assert len(out) == 1

    def test_ツール結果は_role_tool_と_tool_call_id_で返す(self):
        call = ToolCall(id="c1", name="get_volume", args={})
        out = to_openai_messages([
            Message(role="user", text="音量は？"),
            Message(role="assistant", tool_calls=[call]),
            Message(role="tool", tool_call_id="c1", text="30%です"),
        ])
        assert out[1]["role"] == "assistant"
        assert out[1]["tool_calls"][0]["id"] == "c1"
        assert out[1]["tool_calls"][0]["function"]["name"] == "get_volume"
        assert out[2] == {"role": "tool", "tool_call_id": "c1", "content": "30%です"}

    def test_引数は_JSON_文字列にして渡す(self):
        call = ToolCall(id="c1", name="open_app", args={"name": "メモ帳"})
        out = to_openai_messages([Message(role="assistant", tool_calls=[call])])
        raw = out[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(raw, str)
        assert json.loads(raw) == {"name": "メモ帳"}

    def test_日本語をエスケープしない(self):
        """ensure_ascii=True だとトークンを無駄に食う。"""
        call = ToolCall(id="c1", name="open_app", args={"name": "メモ帳"})
        out = to_openai_messages([Message(role="assistant", tool_calls=[call])])
        assert "メモ帳" in out[0]["tool_calls"][0]["function"]["arguments"]

    def test_テキスト無しのツール呼び出しは_content_null(self):
        """content を省略すると 400 を返すプロバイダがある。"""
        call = ToolCall(id="c1", name="get_volume", args={})
        out = to_openai_messages([Message(role="assistant", tool_calls=[call])])
        assert "content" in out[0]
        assert out[0]["content"] is None

    def test_role_tool_に_id_が無ければ作れない(self):
        with pytest.raises(ValueError):
            Message(role="tool", text="結果")


class TestToolConversion:
    def test_function_型で包む(self):
        out = to_openai_tools([{
            "name": "get_weather",
            "description": "天気を取得する",
            "parameters": {"type": "object", "properties": {}},
        }])
        assert out[0]["type"] == "function"
        assert out[0]["function"]["name"] == "get_weather"

    def test_parameters_が無くても空スキーマを入れる(self):
        out = to_openai_tools([{"name": "f", "description": "d"}])
        assert out[0]["function"]["parameters"]["type"] == "object"
