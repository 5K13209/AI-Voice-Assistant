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

# 「ヘッドセット」を落とさないこと。ASCII の headset だけ入れていたため、
# 日本語名で出るヘッドセットのマイクが一致せず、OS デフォルトへ落ちていた。
AUDIO_INPUT_KEYWORDS = ["microphone", "mic", "headset", "マイク", "ヘッドセット"]
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

# 登録で録る本数。5 本から 2 本に減らしてある。声紋の質は「本数」より
# 「入っている声の総量」で決まるので、本数ではなく 1 本の長さで稼ぐ。
VOICE_ENROLL_SAMPLES = 2

# 1 本の長さは固定しない。「以上」と言うか、キーを押すまで録り続ける。
# 固定秒数だと、言い終わっていないのに切られるか、言い終わったのに黙って
# 待たされるかのどちらかになる。自己紹介の長さは人によって違う。
#
# 下限は声紋に必要な量の確保。これを超えるまでは終了語を探し始めない
# （「以上」だけ言って終わられると声紋が作れない）。
VOICE_ENROLL_MIN_SECONDS = 4.0

# 上限は暴走防止。終了語もキーも来なかった場合にここで打ち切る。
VOICE_ENROLL_MAX_SECONDS = 60.0

# 終了語を探すために、末尾を再デコードする間隔（秒）。
VOICE_ENROLL_CHECK_INTERVAL = 1.2

# 終了語の判定に使う末尾の長さ（秒）。全体を毎回デコードする必要はない。
VOICE_ENROLL_TAIL_SECONDS = 4.0

# 終了語の直後に付きうる丁寧形。判定の前に落とす。これを見ないと
# 「以上です」が「以上」に一致せず、いつまでも録音が止まらない。
VOICE_ENROLL_POLITE_TAILS = ("ですね", "でした", "です")

# 録音を終える合図。末尾がこれで終わっていたら止める。
VOICE_ENROLL_STOP_WORDS = (
    "以上", "いじょう", "終わり", "おわり", "終わりです",
    "終了", "しゅうりょう", "オッケー", "オーケー",
)

# 照合を通さず全発話を受け付ける（デバッグ用）。
VOICE_AUTH_ENABLED = os.getenv("TOKA_VOICE_AUTH", "1") != "0"

# =========================
# ▼ LLM
# =========================
# プロバイダは環境変数で切り替える。プリセットの一覧と各社の無料枠は
# `python -m toka.llm --list` で見られる。
#
#   ollama    ローカル。回数無制限（既定）
#   lmstudio  ローカル。回数無制限
#   cerebras  無料枠 30 RPM / 1M tok/日
#   groq      無料枠 30 RPM / 14,400 req/日
#   gemini    無料枠 5 RPM
#
# Gemini 無料枠の 5 RPM は、1 リクエストあたり 13 秒の間隔を意味する。
# 会話としては成立しないので、既定をローカルにした。
LLM_PROVIDER = os.getenv("TOKA_LLM_PROVIDER", "ollama")

# 裏方（感情推定・エピソード要約・検索結果の要約）に使うプロバイダ。
# 未指定なら主応答と同じものを兼用する。ローカルなら回数を気にしなくてよい。
LLM_SUB_PROVIDER = os.getenv("TOKA_LLM_SUB_PROVIDER") or None

# 主応答が落ちたときの逃げ先。未指定なら作らない。
LLM_FALLBACK_PROVIDER = os.getenv("TOKA_LLM_FALLBACK_PROVIDER") or None

TEMPERATURE = 0.8

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
# ▼ 話し方のモード
# =========================
# 「まじめに」「アシスタントモード」と言われたら正確さ優先、
# 「会話モード」で普段に戻る。切り替えは services/persona.py。
#
# 正直度  100 は「知らないことは知らないと言う。推測で埋めない」。
#         下げると、あやふやなことも会話の流れで言い切るようになる。
# ユーモア 高いほど冗談・軽口が増える。
#
# どちらのモードでも、「使っていないツールを使ったふりをする」ことは
# 禁止している（PERSONA 側の絶対規則）。正直度で緩めていい部分ではない。
CONVERSATION_HONESTY = 90
CONVERSATION_HUMOR = 55

ASSISTANT_HONESTY = 100
ASSISTANT_HUMOR = 10

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
