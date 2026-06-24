import io
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr

from faster_whisper import WhisperModel

from config import AUDIO_INPUT_KEYWORDS,AUDIO_OUTPUT_KEYWORDS,SPEAKER_ID,TEMP_USER_INPUT
    
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

recognizer = sr.Recognizer()

def auto_select_devices():
    devices = sd.query_devices()
        
    input_id = None
    output_id = None
        
        # 優先したいキーワード（環境に合わせて調整）
    input_keywords = AUDIO_INPUT_KEYWORDS
    output_keywords = AUDIO_OUTPUT_KEYWORDS
        
        # 1. 入力デバイスの自動探索（マイクチャンネル数が1以上、かつキーワードにマッチ）
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            dev_name = dev['name'].lower()

            if any(kw in dev_name for kw in input_keywords):
                input_id = i
                return input_id
                   # 最初に見つかった最適なものを採用
                    
        # 2. 出力デバイスの自動探索（出力チャンネル数が1以上、かつキーワードにマッチ）
    for i, dev in enumerate(devices):
        if dev['max_output_channels'] > 0:
            dev_name = dev['name'].lower()

            if any(kw in dev_name for kw in output_keywords):
                output_id = i
                return output_id
    
        # マッチするものがなかった場合は、OSのデフォルト（None）にフォールバック
    if input_id is None:
        print("最適な入力デバイスが見つからないため、OSデフォルトを使用します。")
    if output_id is None:
        print("最適な出力デバイスが見つからないため、OSデフォルトを使用します。")
            
        # sounddeviceのデフォルト設定を上書き（Noneの場合はOSデフォルトが維持される）
    sd.default.device = (input_id, output_id)
    print(f"設定されたデバイスID -> 入力: {input_id}, 出力: {output_id}")

auto_select_devices()
# =========================
# ▼ 音声（VOICEVOX）
# =========================
class AIVoice:
    @staticmethod
    def speak(text):
        print(f"トーカ:{text}")
        try:
            q = requests.post(
                "http://127.0.0.1:50021/audio_query",
                params={"text": text, "speaker": SPEAKER_ID}
            )
            s = requests.post(
                "http://127.0.0.1:50021/synthesis",
                params={"speaker": SPEAKER_ID},
                json=q.json()
            )

            byte_stream = io.BytesIO(s.content)
        
            # soundfileでメモリ上のWAVから数値配列とサンプリングレートを抽出
            data, samplerate = sf.read(byte_stream)
        
            # 再生(割り込みできるようにそのうち設計したい)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print("メモリからの音声再生に失敗:", e)

# =========================
# ▼ 音声認識
# =========================
class ListenUser:
    def __init__(self):
            with sr.Microphone() as source:
                print("ノイズ調整中...")
                recognizer.adjust_for_ambient_noise(source,duration=1)

    def listen(self):
        try:
            with sr.Microphone() as source:
                print("\n話してください...（待機中）")
                recognizer.dynamic_energy_threshold = False

                try:
                    audio = recognizer.listen(source, phrase_time_limit=10.0)
                except sr.WaitTimeoutError:
                    return None

            raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
            audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

                        # 一時保存
            with open(TEMP_USER_INPUT, "wb") as f:
                f.write(audio.get_wav_data())

            print("解析中...")

            # ▼ ノイズ除去（VAD付き）
            segments, info = whisper_model.transcribe(
                audio_array,
                language="ja",
                vad_filter=True,
                vad_parameters=dict(
                    min_speech_duration_ms=500  # ←短いノイズをカット
                ),
                condition_on_previous_text=False
            )

            user_text = "".join(segment.text for segment in segments).strip()

            if not user_text:
                return None

            user_text = user_text.strip()

            # ▼ 動画系ノイズ除去
            ignore_phrases = ["ご視聴", "チャンネル登録", "字幕"]
            if any(p in user_text for p in ignore_phrases):
                return None

            print(f"あなた: {user_text}")
            return user_text

        except Exception as e:
            print(f"音声認識エラー: {e}")
            return None
