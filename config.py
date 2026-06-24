TEMP_DIR = "temp_file"
TEMP_USER_INPUT = f"{TEMP_DIR}/user_input.wav"

SPEAKER_ID = 6            # 最初から整数(int)
THRESHOLD = 0.3          # 最初から浮動小数点(float)

AUDIO_INPUT_KEYWORDS = ["microphone", "mic", "headset", "マイク"]
AUDIO_OUTPUT_KEYWORDS = ["speaker", "headset", "headphones", "スピーカー", "ヘッドホン"]

DEFAULT_EMOTION = {
    "like": 50,
    "fun": 50,
    "anger": 50,
    "sad": 50,
    "trust": 50
}