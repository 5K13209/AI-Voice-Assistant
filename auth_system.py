import speech_recognition as sr
import os
import sounddevice as sd
import soundfile as sf

from speechbrain.inference.speaker import SpeakerRecognition
from config import THRESHOLD
    
Threshold = THRESHOLD

class RegisterVoice:
    def __init__(self):
        self.recognizer = sr.Recognizer()

#　録音部分
    def record_voice(self, output_path, duration=5, samplerate=16000): 
        print("録音開始...")
    
        # channels=1 でモノラル録音。dtype='float32' が音声処理ライブラリと相性が良い
        myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    
        # 録音終了までメインスレッドをブロックして待つ
        sd.wait()
        print("録音終了。")
    
        # 必要に応じてWAVファイルとして保存
        sf.write(output_path, myrecording, samplerate)

#　回数指定と返却
    def register(self):
        os.makedirs("voice_data", exist_ok=True)

        voice_refs = []

        for i in range(5):

            print(f"{i+1}/5 回目")

            filename = os.path.join(
            "voice_data",
            f"voice_{i}.wav"
        )

            result = self.record_voice(filename)

            if result:
                voice_refs.append(result)

        print(voice_refs)

        return voice_refs
    
#==========================
# ▼ 声帯認証
# =========================
class VoiceAuth:
    def __init__(self,memory_manager):
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        ) 

        self.memory_manager = memory_manager

    def voice_check(self, path):
        refs = self.memory_manager.get_voice_refs()

        print(path)
        print(os.path.exists(path))

        if not refs:
            print("登録音声がありません")
            return False

        scores = []

        for ref in refs:

            if not os.path.exists(ref):
                print(f"ファイルなし: {ref}")
                continue

            print(ref)
            print(os.path.exists(ref))
        
            score, _ = self.model.verify_files(ref, path)

            print(f"{ref} : {float(score):.4f}")

            scores.append(float(score))

        if len(scores) == 0:
            print("利用可能な登録音声がありません")
            return False

        max_score = max(scores)

        print(f"平均スコア: {max_score:.3f}")

        if max_score > Threshold:
            print("本人")
            return True

        print("他人")
        return False
