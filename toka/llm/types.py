"""プロバイダ非依存の会話表現。

旧実装は会話履歴をそのまま google.genai の types.Content で持っていた。
そのためプロバイダを替えると履歴の型ごと総取り替えになり、割り込み時の
履歴整形（llm.py の note_interruption など）も Gemini の作法に縛られていた。

ここで定義するのは「どのプロバイダでも表せる最小の共通形」だけである。
各プロバイダ固有の作法は adapter 側へ閉じ込める。具体的には:

* Gemini はツール結果を role="user" の function_response として返すが、
  OpenAI 互換は role="tool" + tool_call_id で返す。ここでは role="tool" と
  tool_call_id を持つ形に正規化し、Gemini adapter が変換する。
* Gemini は system をリクエスト config のフィールドに載せるが、OpenAI 互換は
  messages[0] の role="system" として送る。so system はここでは履歴に
  入れず、stream() の引数として別に渡す。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]


@dataclass
class ToolCall:
    """LLM からのツール呼び出し 1 件。

    id は OpenAI 互換の tool_call_id に対応する。Gemini は id を持たないので
    adapter 側で採番する（履歴の突き合わせにこちらでも id が必要なため）。
    """

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """履歴 1 件。

    role="assistant" のとき、text と tool_calls は同時に入りうる
    （何か喋ってからツールを呼ぶケース）。
    role="tool" のときは tool_call_id が必須で、text に実行結果が入る。
    """

    role: Role
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("role='tool' には tool_call_id が必要です")


@dataclass
class Delta:
    """ストリーミングの 1 差分。

    text と tool_call のどちらか一方が入る。ツール呼び出しは断片で届くので、
    adapter 側で 1 件ぶん組み立て終えてから、完成した ToolCall として
    一度だけ流す（呼び出し側に再組み立ての責務を持たせない）。
    """

    text: str | None = None
    tool_call: ToolCall | None = None


class BackendError(RuntimeError):
    """バックエンドが応答を返せなかった。

    retryable=True は「時間を置けば直る見込み」（429・5xx・接続断）を意味し、
    router がフォールバック先へ回すかどうかの判断に使う。
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status
