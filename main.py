import sys
import re
import time
import os
import traceback

from google import genai
from google.genai import types
from faster_whisper import WhisperModel

from dotenv import load_dotenv

from auth_system import RegisterVoice,VoiceAuth
from memory_manager import MemoryManager
from audio_handlers import ListenUser , AIVoice,auto_select_devices
from logic_utils import Emotion, AppUtils
from config import DEFAULT_EMOTION,TEMP_USER_INPUT

# --- 文字化け対策 ---
sys.stdout.reconfigure(encoding='utf-8')

sys.stderr.reconfigure(encoding='utf-8')

# --- API設定 ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# --- Whisper ---
print("--- 起動 ---")
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

# --- モデル選択 ---
def select_model():
    while True:
        lm = input('モデル選択(f:Flash, p:Pro) > ')
        if lm == 'f':
            return "gemini-2.5-flash"
        elif lm == 'p':
            return "gemini-2.5-pro"

MODEL_NAME = select_model()

# =========================
# ▼ メイン
# =========================
memory_manager = MemoryManager()

if "emotion" in memory_manager.memory:
    # あれば、前回のデータを引き継ぐ
    toka_emotion = memory_manager.memory["emotion"]
else:
    # なければ、初期値（オール50）をセットして保存する（初回起動時のみここを通る）
    toka_emotion = DEFAULT_EMOTION
    memory_manager.memory["toka_emotion"] = toka_emotion
    memory_manager.save_memory()

config = types.GenerateContentConfig(
    system_instruction=f'あなたはトーカ。基本は冷静。現在の感情:好感度:{toka_emotion["like"]}怒り:{toka_emotion["anger"]}楽しさ:{toka_emotion["fun"]}信頼:{toka_emotion["trust"]}感情に応じて自然に会話してください.audio_timestamp=EMOTIONの各値は50が基準であり、それを下回ればマイナスの感情、超えるとプラスの感情。また最低0、最高を100とし、その値に適した形でロールプレイをして',
    temperature=0.8
)    

chat = client.chats.create(model=MODEL_NAME, config=config)

Vtest = False

voice_auth = VoiceAuth(memory_manager)
voice_register = RegisterVoice()

audio_check = auto_select_devices()
ai_voice = AIVoice()
listen_user = ListenUser()


refs = memory_manager.get_voice_refs()

last_request_time = 0

print("memory refs =", memory_manager.get_voice_refs())

for ref in memory_manager.get_voice_refs():
    print(ref, os.path.exists(ref))

ai_voice.speak("起動完了。")

while True:
    need_register = False

    if len(refs) == 0:
        need_register = True

    for ref in refs:
        if not os.path.exists(ref):
            need_register = True
            break

    if need_register:
        print("声紋登録を開始します")
        
        refs = voice_register.register()

        memory_manager.memory["voice_refs"] = refs
        memory_manager.save_memory()

        print("登録完了")

    user_input = listen_user.listen()
    
    if not user_input:
        print("① listen not OK")
        continue
    print("① listen成功")
   
    Vtest = voice_auth.voice_check(TEMP_USER_INPUT)

    print("認証結果 =", Vtest)

    if not Vtest:
        continue
    
    try:
        # 感情更新
        toka_emotion = Emotion.update_emotion(user_input, toka_emotion)
        
        # 429エラー対策
        if(time.time() - last_request_time <= 5):
            lest_time = AppUtils.can_send()
            print("API制限を考慮して待機中...")

        # 記憶保存（ユーザー）
        memory_manager.store_memory(f"ユーザー: {user_input}", toka_emotion)

        # プロンプト生成
        prompt = memory_manager.build_prompt(user_input, toka_emotion)
        print("③ Gemini送信")
        # LLM
        res = chat.send_message(prompt)
        last_request_time = time.time()

        text = re.sub(r'[*_`#]', '', res.text)
        print("④ Gemini受信")
        
        # 記憶保存（AI）
        memory_manager.store_memory(f"AI: {text}", toka_emotion)

        ai_voice.speak(text)

    except Exception as e:
        traceback.print_exc()