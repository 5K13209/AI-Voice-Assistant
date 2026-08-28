"""ツールから本体の機能へ触るための受け渡し口。

ツールはただの関数なので、記憶やバスへの参照を引数で受け取れない
（引数はすべて LLM が埋めるため）。runtime が起動時にここへ差し込む。

グローバル状態だが、プロセス内に 1 つしか存在しないものへの参照であり、
これを避けようとすると全ツールをクラス化してファクトリを通す必要があって
記述量に見合わない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..bus import EventBus
    from ..services.memory import MemoryManager


@dataclass
class ToolContext:
    bus: EventBus | None = None
    memory: MemoryManager | None = None
    genai_client: Any | None = None
    # 自発発話のオン・オフ。ユーザーが「黙ってて」と言えるようにする。
    proactive_enabled: bool = True


CONTEXT = ToolContext()


def bind(bus, memory, genai_client) -> None:
    CONTEXT.bus = bus
    CONTEXT.memory = memory
    CONTEXT.genai_client = genai_client
