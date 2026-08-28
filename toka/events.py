"""イベント定義。

各サービスが互いを直接呼ばずに済むよう、やり取りはすべてここに定義した
dataclass を通す。将来 UI を足すときも、UI はバスからこれらを購読するだけで
サービス側には一切手を入れなくてよい。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class State(str, Enum):
    """runtime の状態機械。"""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass
class Event:
    """全イベントの基底。発生時刻だけ共通で持つ。"""

    at: float = field(default_factory=time.time, init=False, repr=False)


# =========================
# ▼ 音声入力
# =========================


@dataclass
class SpeechDetected(Event):
    """VAD が発話の始まりを検出した。まだ内容は分からない。"""


@dataclass
class PartialTranscript(Event):
    """発話途中の暫定認識結果。確定ではないので履歴には残さない。"""

    text: str


@dataclass
class Utterance(Event):
    """発話区間が閉じ、認識が確定した。話者照合はまだ通っていない。"""

    text: str
    wav_path: str
    duration: float


@dataclass
class SpeakerRejected(Event):
    """話者照合に落ちた発話。TTS の回り込みもここで落ちる。"""

    text: str
    score: float


@dataclass
class UserMessage(Event):
    """照合を通り、LLM に渡してよいと確定したユーザー発話。"""

    text: str
    score: float


# =========================
# ▼ 応答
# =========================


@dataclass
class ThinkingStarted(Event):
    """LLM への送信を開始した。"""

    trigger: str = "user"  # "user" | "proactive" | "timer"


@dataclass
class AssistantDelta(Event):
    """LLM のストリーミング差分。UI 用。TTS は文単位の方を使う。"""

    text: str


@dataclass
class AssistantSentence(Event):
    """句点で区切れた 1 文。これが TTS の投入単位になる。"""

    text: str


@dataclass
class AssistantMessage(Event):
    """1 応答の確定。割り込まれた場合は interrupted=True で途中まで。"""

    text: str
    interrupted: bool = False


# =========================
# ▼ 発話（出力）
# =========================


@dataclass
class SpeechStarted(Event):
    """TTS が実際に音を出し始めた。"""

    text: str


@dataclass
class SpeechFinished(Event):
    """1 文の再生が最後まで終わった。"""

    text: str


@dataclass
class BargeIn(Event):
    """発話中にユーザーが割り込んだ。再生と生成の両方を止める。"""

    reason: str = "user_speech"


# =========================
# ▼ ツール
# =========================


@dataclass
class ToolCalled(Event):
    """LLM がツールを呼んだ。"""

    name: str
    args: dict[str, Any]


@dataclass
class ToolResult(Event):
    """ツールの実行結果。ok=False なら result にエラー文が入る。"""

    name: str
    result: str
    ok: bool = True


@dataclass
class ToolConfirmationNeeded(Event):
    """risk="confirm" のツール。ユーザーの同意を待っている。"""

    name: str
    args: dict[str, Any]
    question: str


# =========================
# ▼ 状態・感情・記憶
# =========================


@dataclass
class StateChanged(Event):
    old: State
    new: State


@dataclass
class EmotionChanged(Event):
    emotion: dict[str, int]
    delta: dict[str, int]


@dataclass
class MemoryStored(Event):
    kind: str  # "event" | "episode" | "profile"
    text: str


# =========================
# ▼ 自発
# =========================


@dataclass
class ProactiveImpulse(Event):
    """「今声をかけたい」という衝動。LLM が黙る判断をすることもある。"""

    kind: str  # "idle" | "clock" | "window" | "timer"
    context: str


@dataclass
class TimerFired(Event):
    """set_timer ツールで仕掛けたタイマーの発火。"""

    label: str


# =========================
# ▼ ライフサイクル
# =========================


@dataclass
class Started(Event):
    pass


@dataclass
class ShuttingDown(Event):
    pass


@dataclass
class ServiceFailed(Event):
    """サービスのタスクが例外で落ちた。旧実装の裸 except の代わり。"""

    service: str
    error: str
