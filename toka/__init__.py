"""トーカ — PCの中に住む音声アシスタント。

構成:

    bus.py        イベントバス。サービス同士はこれ越しにしか繋がらない
    events.py     イベント定義。サービス間の唯一の共通言語
    runtime.py    起動・配線・状態遷移
    services/     capture, stt, auth, llm, tts, memory, emotion, proactive
    tools/        LLM から呼べる PC 操作

画面出力を足すときは、バスを購読するタスクを 1 本増やすだけでよい。
既存のサービスには手を入れない。
"""

__version__ = "0.2.0"
