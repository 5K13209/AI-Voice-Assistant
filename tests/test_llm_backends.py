"""バックエンド共通部分（スキーマ変換・レート制限・ルータ）の単体テスト。"""

from __future__ import annotations

import asyncio

import pytest

from toka.llm.gemini import resolve_tool_name, retry_after, to_gemini_schema
from toka.llm.limiter import RateLimiter
from toka.llm.router import LLMRouter
from toka.llm.types import BackendError, Delta, Message, ToolCall


class TestGeminiSchema:
    def test_型名を大文字にする(self):
        out = to_gemini_schema({"type": "object", "properties": {
            "n": {"type": "integer"}, "s": {"type": "string"},
        }})
        assert out["type"] == "OBJECT"
        assert out["properties"]["n"]["type"] == "INTEGER"
        assert out["properties"]["s"]["type"] == "STRING"

    def test_入れ子も変換する(self):
        out = to_gemini_schema({
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "number"}}},
        })
        assert out["properties"]["items"]["type"] == "ARRAY"
        assert out["properties"]["items"]["items"]["type"] == "NUMBER"

    def test_description_は触らない(self):
        out = to_gemini_schema({"type": "string", "description": "都市名"})
        assert out["description"] == "都市名"

    def test_properties_のキーを型名と間違えない(self):
        """"type" という名前のプロパティがあっても壊れないこと。"""
        out = to_gemini_schema({
            "type": "object",
            "properties": {"type": {"type": "string"}},
        })
        assert out["properties"]["type"]["type"] == "STRING"

    def test_required_はそのまま(self):
        out = to_gemini_schema({"type": "object", "properties": {},
                                "required": ["a", "b"]})
        assert out["required"] == ["a", "b"]


class TestRetryAfter:
    def test_retryDelay_を秒で取り出す(self):
        exc = Exception()
        exc.details = {"error": {"details": [
            {"@type": "type.googleapis.com/google.rpc.RetryInfo",
             "retryDelay": "41s"},
        ]}}
        assert retry_after(exc) == 41.0

    def test_details_が無ければ_None(self):
        assert retry_after(Exception()) is None

    def test_形が違えば_None(self):
        exc = Exception()
        exc.details = "not a dict"
        assert retry_after(exc) is None

    def test_retryDelay_が壊れていれば_None(self):
        exc = Exception()
        exc.details = {"error": {"details": [{"retryDelay": "abcs"}]}}
        assert retry_after(exc) is None


class TestResolveToolName:
    def test_直前の呼び出しから名前を引く(self):
        """Gemini は id ではなく関数名で紐付けるため、逆引きが必要。"""
        messages = [
            Message(role="user", text="音量は？"),
            Message(role="assistant",
                    tool_calls=[ToolCall(id="c1", name="get_volume")]),
            Message(role="tool", tool_call_id="c1", text="30%"),
        ]
        assert resolve_tool_name(messages, 2) == "get_volume"

    def test_複数の候補から正しいものを選ぶ(self):
        messages = [
            Message(role="assistant", tool_calls=[
                ToolCall(id="c1", name="a"), ToolCall(id="c2", name="b"),
            ]),
            Message(role="tool", tool_call_id="c2", text="r"),
        ]
        assert resolve_tool_name(messages, 1) == "b"

    def test_見つからなければ既定値(self):
        messages = [Message(role="tool", tool_call_id="missing", text="r")]
        assert resolve_tool_name(messages, 0) == "unknown_tool"


class TestRateLimiter:
    def test_rpm_0_は無制限(self):
        limiter = RateLimiter(0)
        assert limiter.unlimited

    @pytest.mark.asyncio
    async def test_無制限なら待たない(self):
        limiter = RateLimiter(0)
        started = asyncio.get_event_loop().time()
        for _ in range(5):
            await limiter.acquire()
        assert asyncio.get_event_loop().time() - started < 0.1

    def test_rpm_から間隔を決める(self):
        # 30 RPM なら 2 秒間隔 + 1 割の余裕。
        limiter = RateLimiter(30)
        assert not limiter.unlimited
        assert 2.0 < limiter._min_interval < 2.5

    @pytest.mark.asyncio
    async def test_初回は待たない(self):
        limiter = RateLimiter(30)
        started = asyncio.get_event_loop().time()
        await limiter.acquire()
        assert asyncio.get_event_loop().time() - started < 0.1

    @pytest.mark.asyncio
    async def test_penalize_は無制限でも効く(self):
        """ローカルでも 503 は起こりうるので、指定された待ちには従う。"""
        limiter = RateLimiter(0)
        limiter.penalize(0.2)
        started = asyncio.get_event_loop().time()
        await limiter.acquire()
        assert asyncio.get_event_loop().time() - started >= 0.15

    def test_負の_penalize_は無視する(self):
        limiter = RateLimiter(0)
        limiter.penalize(-5)
        assert limiter._until == 0.0


class _FakeBackend:
    """指定した回数だけ失敗してから成功するバックエンド。"""

    def __init__(self, name, *, fail=0, retryable=True, deltas=None,
                 fail_after_first=False):
        self.name = name
        self.rpm = 0
        self.calls = 0
        self._fail = fail
        self._retryable = retryable
        self._deltas = deltas or [Delta(text="ok")]
        self._fail_after_first = fail_after_first

    async def stream(self, messages, *, system="", tools=None, temperature=0.8):
        self.calls += 1
        if self.calls <= self._fail:
            raise BackendError(f"{self.name} 失敗", retryable=self._retryable)
        for delta in self._deltas:
            yield delta
            if self._fail_after_first:
                raise BackendError("途中で失敗", retryable=True)

    async def complete(self, prompt, *, schema=None, temperature=0.0):
        self.calls += 1
        if self.calls <= self._fail:
            raise BackendError(f"{self.name} 失敗", retryable=self._retryable)
        return "完了"

    async def healthy(self):
        return True

    async def close(self):
        return None


async def _collect(router, **kwargs):
    return [d async for d in router.stream([Message(role="user", text="x")], **kwargs)]


class TestRouter:
    @pytest.mark.asyncio
    async def test_main_が通れば_fallback_を使わない(self):
        main = _FakeBackend("main")
        fallback = _FakeBackend("fb")
        router = LLMRouter(main, fallback=fallback)
        assert [d.text for d in await _collect(router)] == ["ok"]
        assert fallback.calls == 0

    @pytest.mark.asyncio
    async def test_一時的な失敗は再試行する(self):
        main = _FakeBackend("main", fail=1)
        router = LLMRouter(main)
        assert [d.text for d in await _collect(router)] == ["ok"]
        assert main.calls == 2

    @pytest.mark.asyncio
    async def test_恒久的な失敗は再試行しない(self):
        main = _FakeBackend("main", fail=99, retryable=False)
        router = LLMRouter(main)
        with pytest.raises(BackendError):
            await _collect(router)
        assert main.calls == 1

    @pytest.mark.asyncio
    async def test_諦めたら_fallback_へ回す(self):
        main = _FakeBackend("main", fail=99)
        fallback = _FakeBackend("fb", deltas=[Delta(text="代替")])
        router = LLMRouter(main, fallback=fallback)
        assert [d.text for d in await _collect(router)] == ["代替"]

    @pytest.mark.asyncio
    async def test_恒久的な失敗では_fallback_へ回さない(self):
        """400 や認証エラーで逃げても同じ結果になるだけ。"""
        main = _FakeBackend("main", fail=99, retryable=False)
        fallback = _FakeBackend("fb")
        router = LLMRouter(main, fallback=fallback)
        with pytest.raises(BackendError):
            await _collect(router)
        assert fallback.calls == 0

    @pytest.mark.asyncio
    async def test_流し始めた後は切り替えない(self):
        """既に喋っているので、やり直すと同じ内容を二度読み上げる。"""
        main = _FakeBackend("main", deltas=[Delta(text="途中まで")],
                            fail_after_first=True)
        fallback = _FakeBackend("fb")
        router = LLMRouter(main, fallback=fallback)
        with pytest.raises(BackendError):
            await _collect(router)
        assert fallback.calls == 0
        assert main.calls == 1

    @pytest.mark.asyncio
    async def test_sub_を省略すると_main_を兼用する(self):
        main = _FakeBackend("main")
        router = LLMRouter(main)
        assert router.sub is main
        assert await router.complete_sub("x") == "完了"

    @pytest.mark.asyncio
    async def test_sub_の失敗も_fallback_で拾う(self):
        sub = _FakeBackend("sub", fail=99)
        fallback = _FakeBackend("fb")
        router = LLMRouter(_FakeBackend("main"), sub=sub, fallback=fallback)
        assert await router.complete_sub("x") == "完了"

    def test_describe_に構成が出る(self):
        router = LLMRouter(_FakeBackend("main"), fallback=_FakeBackend("fb"))
        text = router.describe()
        assert "main" in text and "fb" in text
