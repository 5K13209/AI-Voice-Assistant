"""ツール登録。Python 関数から Gemini の FunctionDeclaration を組み立てる。

    @tool(risk="safe", params={"name": "アプリの通称"})
    def open_app(name: str) -> str:
        '''許可されたアプリケーションを起動する。'''

google-genai には Python 関数をそのまま渡す自動関数呼び出しがあるが、使わない。
確認ゲートを挟みたいこと、asyncio 上で実行したいこと、割り込みでキャンセル
できる必要があること、の 3 つが理由。呼び出しループは llm.py 側に持つ。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger(__name__)

Risk = Literal["safe", "confirm"]

_PY_TO_GEMINI = {
    str: "STRING",
    int: "INTEGER",
    float: "NUMBER",
    bool: "BOOLEAN",
    list: "ARRAY",
    dict: "OBJECT",
}


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    risk: Risk = "safe"
    params: dict[str, str] = field(default_factory=dict)
    confirm_template: str = "{name} を実行していい？"

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.func)

    def question(self, args: dict[str, Any]) -> str:
        """risk="confirm" のとき、ユーザーに読み上げる確認文。"""
        detail = "、".join(f"{k}={v}" for k, v in args.items()) or "引数なし"
        return self.confirm_template.format(name=self.name, args=detail)

    def declaration(self) -> dict[str, Any]:
        """Gemini の FunctionDeclaration に渡す dict を作る。"""
        signature = inspect.signature(self.func)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in signature.parameters.items():
            annotation = param.annotation
            # `str | None` のような Union は最初の実型を採用する。
            origin_args = getattr(annotation, "__args__", None)
            if origin_args:
                annotation = next(
                    (a for a in origin_args if a is not type(None)), str
                )

            properties[param_name] = {
                "type": _PY_TO_GEMINI.get(annotation, "STRING"),
                "description": self.params.get(param_name, param_name),
            }
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema: dict[str, Any] = {"type": "OBJECT", "properties": properties}
        if required:
            schema["required"] = required

        return {
            "name": self.name,
            "description": self.description,
            "parameters": schema,
        }

    async def call(self, args: dict[str, Any], timeout: float) -> str:
        """ツールを実行する。同期関数は executor に逃がす。"""
        if self.is_async:
            coro = self.func(**args)
        else:
            loop = asyncio.get_running_loop()
            coro = loop.run_in_executor(None, lambda: self.func(**args))
        result = await asyncio.wait_for(coro, timeout)
        return str(result)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"ツール名が重複しています: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def declarations(self) -> list[dict[str, Any]]:
        return [tool.declaration() for tool in self._tools.values()]


# モジュール読み込み時に @tool が積むグローバル登録簿。
REGISTRY = ToolRegistry()


def tool(
    *,
    risk: Risk = "safe",
    name: str | None = None,
    params: dict[str, str] | None = None,
    confirm: str = "{name} を実行していい？",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """関数をツールとして登録するデコレータ。

    risk="safe"    無確認で実行する。読み取りと可逆な操作のみ。
    risk="confirm" 実行前に音声で確認を取る。外に影響が出る操作。

    危険なもの（ファイル削除、任意のシェル実行）はそもそも登録しない。
    「拒否リスト」ではなく「登録しない」で担保する。
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        doc = inspect.getdoc(func) or ""
        # docstring の 1 行目までを説明にする。以降は実装メモとして扱う。
        description = doc.split("\n\n")[0].strip() or func.__name__
        REGISTRY.register(
            Tool(
                name=name or func.__name__,
                description=description,
                func=func,
                risk=risk,
                params=params or {},
                confirm_template=confirm,
            )
        )
        return func

    return decorator


def load_all() -> ToolRegistry:
    """ツールモジュールを import して REGISTRY を埋める。"""
    from . import files, inner, system, web  # noqa: F401

    log.info("ツール %d 件を登録: %s", len(REGISTRY), ", ".join(REGISTRY.names()))
    return REGISTRY
