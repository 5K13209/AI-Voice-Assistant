"""設定値の一元管理。

環境変数で上書きできるものは os.getenv 経由にしてある。それ以外は
ここを直接書き換える。旧 config.py の定数はすべてここへ移設済み。
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# =========================
# ▼ パス
# =========================
MODELS_DIR = ROOT / "models"
TEMP_DIR = ROOT / "temp_file"
VOICE_DATA_DIR = ROOT / "voice_data"
CHROMA_DIR = ROOT / "chroma_db"
MEMORY_FILE = ROOT / "memory.json"

# 話者照合に渡す発話の WAV は temp_file/utt-<id>.wav として 1 発話 1 ファイルで
# 書き出す（STTService._write_wav）。固定パスに上書きしていた旧実装は、
# 発話が近接したときに照合前の音声を潰していた。

# =========================
# ▼ 音声 I/O
# =========================
SAMPLE_RATE = 16000

# マイクのコールバックが一度に返すフレーム数。20ms 相当。
# 小さいほど割り込みの反応が速いが、コールバック回数が増える。
FRAME_SIZE = 320

AUDIO_INPUT_KEYWORDS = ["microphone", "mic", "headset", "マイク"]
AUDIO_OUTPUT_KEYWORDS = ["speaker", "headset", "headphones", "スピーカー", "ヘッドホン"]

# 明示指定したい場合はデバイス番号を入れる（None なら自動探索）。
AUDIO_INPUT_DEVICE = None
AUDIO_OUTPUT_DEVICE = None

# =========================
# ▼ STT
# =========================
# "sherpa" (ReazonSpeech Zipformer) / "whisper" (旧 faster-whisper 経路)
STT_ENGINE = os.getenv("TOKA_STT_ENGINE", "sherpa")

SHERPA_ASR_DIR = MODELS_DIR / "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17"
SHERPA_VAD_MODEL = MODELS_DIR / "silero_vad.onnx"
SHERPA_NUM_THREADS = 2

WHISPER_MODEL_SIZE = "small"

# VAD。min_silence_duration がそのまま「言い終わってから反応するまで」の
# 体感レイテンシになる。ただし短くしすぎると文の途中の息継ぎで切られ、
# 半分だけ聞いて返事をする。実測では 0.35 秒だと一文が 3〜4 個に割れた。
VAD_THRESHOLD = 0.35
VAD_MIN_SILENCE = 0.7
VAD_MIN_SPEECH = 0.25
VAD_MAX_SPEECH = 20.0

# 切り出した区間の前後に足す音声の長さ（秒）。
# Silero VAD の境界は詰まっていて、語頭の子音や小声の語尾が落ちる。実測で
# 元音声の 58% しか拾えていなかったものが、0.3 秒足すと 75% まで戻った。
VAD_PADDING = 0.3

# パディング用に保持しておく生音声の長さ（秒）。
AUDIO_BUFFER_SECONDS = 30

# 発話中に途中経過を再デコードする間隔（秒）。0 で無効。
PARTIAL_INTERVAL = 0.5

# TTS 再生中は自分の声を拾いやすいので VAD を鈍くする。
VAD_THRESHOLD_WHILE_SPEAKING = 0.75

# =========================
# ▼ TTS (VOICEVOX)
# =========================
VOICEVOX_URL = os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021")
SPEAKER_ID = int(os.getenv("VOICEVOX_SPEAKER_ID", "6"))

# 再生を書き込むチャンク長（秒）。割り込み時はこの粒度で止まる。
PLAYBACK_CHUNK = 0.05

# =========================
# ▼ 話者照合
# =========================
# ECAPA のコサインスコア閾値。旧実装の 0.3 はかなり緩い。
# 自分の環境で voice_check のスコアを見ながら調整すること。
VOICE_THRESHOLD = float(os.getenv("TOKA_VOICE_THRESHOLD", "0.45"))
VOICE_ENROLL_SAMPLES = 5
VOICE_ENROLL_DURATION = 5

# 照合を通さず全発話を受け付ける（デバッグ用）。
VOICE_AUTH_ENABLED = os.getenv("TOKA_VOICE_AUTH", "1") != "0"

# =========================
# ▼ LLM
# =========================
GEMINI_MODEL = os.getenv("TOKA_MODEL", "gemini-2.5-flash")
GEMINI_SUB_MODEL = "gemini-2.5-flash"  # 感情推定・要約などの裏方用
TEMPERATURE = 0.8

# Gemini の 1 分あたりリクエスト上限。無料枠の gemini-2.5-flash は 5。
# 有料枠に上げたらここを上げる。実測で 429 が出るなら下げる。
GEMINI_RPM = int(os.getenv("TOKA_GEMINI_RPM", "5"))

# 自主レート制限。旧実装の can_send() は常に True を返す no-op だった。
# 上限ぴったりだと境界で 429 を踏むので、1 割ほど余裕を持たせる。
MIN_REQUEST_INTERVAL = 60.0 / GEMINI_RPM * 1.1

# 履歴として保持する会話ターン数。超えた分は episodes に要約して落とす。
MAX_HISTORY_TURNS = 20

# =========================
# ▼ 感情
# =========================
DEFAULT_EMOTION = {"like": 50, "fun": 50, "anger": 50, "sad": 50, "trust": 50}

# 1ターンあたりの変動幅の上限。LLM が極端な値を返しても暴れないようにする。
EMOTION_MAX_DELTA = 5

# 感情をどう更新するか。
#   "tool"     応答と同じリクエストの中で feel ツールを呼ばせる（API消費 0 回）
#   "separate" ターンごとに軽量モデルへ別途投げる（精度は上だが 1 回消費する）
#   "keyword"  キーワード加点のみ（API を使わない。旧実装相当）
# 無料枠は 5 リクエスト/分しかないので、既定は "tool"。
EMOTION_MODE = os.getenv("TOKA_EMOTION_MODE", "tool")

# =========================
# ▼ 記憶
# =========================
# 旧 all-MiniLM-L6-v2 は英語専用。日本語会話の想起がほぼ効いていなかった。
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# 埋め込みモデルを変えるとベクトル空間が変わるので、コレクション名も変える。
# 旧 "toka_memory" とは次元が同じため、同居させると静かに壊れる。
CHROMA_COLLECTION = "toka_memory_v2"

MAX_EVENTS = 200
RECALL_TOP_K = 4

# 何ターンごとに会話をエピソードとして要約するか。
EPISODE_EVERY_TURNS = 8

# =========================
# ▼ 自発発話
# =========================
PROACTIVE_ENABLED = os.getenv("TOKA_PROACTIVE", "1") != "0"

# 最終ユーザー発話からこの秒数で「無言」衝動が立つ。
PROACTIVE_IDLE_SECONDS = 10 * 60

# 自発発話同士の最短間隔。無視されるたびに倍化し、上限で頭打ちにする。
PROACTIVE_COOLDOWN = 15 * 60
PROACTIVE_COOLDOWN_MAX = 2 * 60 * 60

# アクティブウィンドウの監視間隔と、同一ウィンドウ滞在の通知しきい値。
WINDOW_POLL_INTERVAL = 5.0
WINDOW_DWELL_SECONDS = 45 * 60

# この時間帯は自発発話しない (start_hour, end_hour)。またぎ可 (23, 8) など。
QUIET_HOURS = (1, 8)

# =========================
# ▼ ツール
# =========================
# 起動を許可するアプリ。通称 -> 実行コマンド。ここに無いものは実行しない。
ALLOWED_APPS = {
    "メモ帳": "notepad.exe",
    "電卓": "calc.exe",
    "エクスプローラ": "explorer.exe",
    "ブラウザ": "https://www.google.com",
    "ペイント": "mspaint.exe",
    "設定": "ms-settings:",
    "タスクマネージャ": "taskmgr.exe",
    "ターミナル": "wt.exe",
    "vscode": "code",
}

# ファイル関連ツールが触れるディレクトリ。これ以外は拒否する。
ALLOWED_FILE_ROOTS = [
    Path.home() / "Documents",
    Path.home() / "Downloads",
    Path.home() / "Desktop",
]

# 1回のツール実行に許す秒数。
TOOL_TIMEOUT = 20.0

# LLM が 1 応答内で連鎖できるツール呼び出しの上限。
MAX_TOOL_ITERATIONS = 5
