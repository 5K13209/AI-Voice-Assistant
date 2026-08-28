# AI Voice Assistant — トーカ

## 1. 概要

「PCの中に住んでいて、こちらが作業している横で自然に話しかけてくる相手」を目指した音声アシスタントです。カゲロウプロジェクトのエネのような存在が最終目標で、そこへ向けたバックエンドの実装がこのリポジトリです。

**主要機能**

* **全二重の音声対話**: マイクは常に開いたまま。喋っている最中でも聞いており、話しかけられれば言葉の途中で黙る。
* **音声生体認証**: 登録された話者の声だけを受け付ける。ウェイクワードは無い。
* **PC操作**: アプリ起動、音量、クリップボード、ファイル検索、Web検索などをツールとして実行する。
* **自発発話**: 話しかけられなくても、無言が続いたときや同じ画面を長時間開いているときに自分から声をかける。
* **感情モデル**: 会話内容に応じて感情が動き、JSONに永続化される。その値は毎ターンのシステムプロンプトに反映される。
* **長期記憶（RAG）**: 会話をエピソード単位で要約し、ベクトル検索で想起する。

---

## 2. 技術スタック

| 用途 | 採用 |
| :--- | :--- |
| LLM | Google Gemini API (`google-genai`) — ストリーミング + function calling |
| 音声認識 (STT) | `sherpa-onnx` 上の **ReazonSpeech k2-v2 Zipformer**（CPU / INT8 / ONNX） |
| 発話区間検出 (VAD) | Silero VAD（sherpa-onnx 同梱） |
| 音声合成 (TTS) | `VOICEVOX`（HTTP API） |
| オーディオI/O | `sounddevice` + `soundfile`（入出力とも統一） |
| 話者認識 | `SpeechBrain` ECAPA-TDNN |
| ベクトルDB | `ChromaDB` |
| 埋め込み | `sentence-transformers` / `paraphrase-multilingual-MiniLM-L12-v2` |

---

## 3. アーキテクチャ

`while True` の手続きループを捨て、**asyncio のイベントバス + 非同期サービス群**で構成しています。各サービスはバスに publish し、必要なイベントを subscribe するだけで、互いを直接呼びません。

```
toka/
  bus.py           イベントバス。サービス同士はこれ越しにしか繋がらない
  events.py        イベント定義。サービス間の唯一の共通言語
  runtime.py       起動・配線・状態遷移 (IDLE / LISTENING / THINKING / SPEAKING)
  config.py        設定値の一元管理
  audio_devices.py 入出力デバイスの選択
  model_files.py   ONNX モデルのファイル名解決
  services/
    capture.py     マイク常時キャプチャ（PortAudio コールバック → asyncio）
    stt.py         VAD + ReazonSpeech 認識、部分認識
    auth.py        ECAPA 話者照合
    llm.py         Gemini ストリーミング、ツールループ、履歴管理、レート制限
    tts.py         VOICEVOX 文単位ストリーミング再生（中断可能）
    memory.py      events / episodes / profile の3層記憶
    emotion.py     感情推定とエピソード要約
    proactive.py   自発発話のトリガ
  tools/
    registry.py    @tool デコレータ → FunctionDeclaration 自動生成
    system.py      アプリ起動 / 音量 / クリップボード / アクティブウィンドウ
    files.py       ファイル検索・読み取り（読み取り専用）
    web.py         Web検索
    inner.py       remember / feel / set_timer / set_proactive
scripts/
  fetch_models.py    ONNX モデルの取得
  migrate_memory.py  旧形式の記憶の移行
main.py            薄いエントリポイント
```

画面出力（Live2D / オーバーレイ / Web UI）を足すときは、**バスを購読するタスクを1本追加するだけ**で、既存サービスには手を入れません。

---

## 4. 技術的なこだわり・設計思想

### 全二重 — 「喋りながら聞く」を成立させる

旧実装では `sd.wait()` と `recognizer.listen()` がどちらもブロッキングで、喋っている間は原理的に耳が塞がっていました。現在は次の3点で割り込みを成立させています。

* **キャプチャを止めない。** `sd.RawInputStream` のコールバックが20msフレームを流し続け、録音の開始・停止という概念自体がありません。
* **部分認識で割り込みを検出する。** 確定認識を待つと言い終わってから1秒近く経ち、割り込みではなく順番待ちになります。発話中は0.5秒ごとに伸長バッファを再デコードし、実測で**発話開始から約0.53秒**で最初の途中経過が出ます。
* **再生を細切れに書く。** `sd.play()` + `sd.wait()` をやめ、`sd.OutputStream` に50msずつ書き込むループにしました。止めたければ次を書かないだけで、**実測62ms**で黙ります。

### ウェイクワードレスと、声紋認証のエコーキャンセラ的な役割

ウェイクワード（「Hey Siri」等）は会話のテンポを阻害すると考え、声紋認証のみで稼働する設計にしています。生体認証単体はリプレイアタックに弱く本来は多要素認証が望ましいのですが、コンセプト実現のため単一認証で割り切っています。

全二重化した今、この層はもうひとつ役目を持ちます。スピーカーから出た自分の声をマイクが拾い戻しても、**VOICEVOXの合成音声はユーザーの声紋（ECAPA x-vector）に一致しないのでここで落ちる**——事実上のエコーキャンセラとして働きます。加えて、再生中はVADの閾値を上げ、割り込み判定では今読み上げている文と部分認識を照合して回り込みを弾いています。

### STT を ReazonSpeech に置き換えた理由と、その限界

同じ音声での実測（CPU）:

| 音声 | ReazonSpeech (sherpa-onnx) | faster-whisper small |
| :--- | :--- | :--- |
| 13秒 | RTF 0.018 | RTF 0.256 |
| 11秒 | RTF 0.017 | RTF 0.271 |
| **1.9秒の実発話** | **0.05秒** | **2.69秒** |

会話用途で効くのは3行目です。短い発話で 0.05秒 と 2.69秒 は「即答」と「気まずい間」の差になります。加えて、**無音や環境音から「ご視聴ありがとうございました」の類をでっち上げません**。旧実装が必要としていた `ignore_phrases` フィルタが構造的に不要になりました。

一方で**精度は一長一短**です。同梱サンプルでは Whisper の方が正しく取れたケースもありますし、ReazonSpeech の精度優位を示すベンチは自身のテスト分割で測ったもので、実環境音声では逆の結果を出す独立系ベンチもあります。`TOKA_STT_ENGINE=whisper` で旧経路に切り替えて比較できるようにしてあります。

### VAD は「測ってから」決める

Silero VAD をそのまま使うと2つ問題が出ました。どちらも実測して設定値を決めています。

* **境界が詰まりすぎて音が落ちる。** 元音声の58%しか拾えず、冒頭の一文が丸ごと消えていました。VADに渡した生音声をリングバッファに保持し、切り出した区間の**前後0.3秒を足して**復元することで75%まで戻ります。
* **`min_silence_duration` 0.35秒では文が割れる。** 一文が3〜4個の断片に分割され、半分だけ聞いて返事をする状態でした。0.7秒に変更しています。これはそのまま「言い終わってから反応するまで」の体感レイテンシになるので、短くしすぎない方が結果的に速く感じます。

### APIレートリミット — 無料枠は5リクエスト/分

Gemini の無料枠（gemini-2.5-flash）は **5リクエスト/分**です。旧実装の3秒間隔は6倍超過していました。現在は `GEMINI_RPM` から間隔を逆算し、429が返ってきた場合はサーバーが指定する `retryDelay` に従います。

この制約のため、**感情の更新は応答と同じリクエストの中で `feel` ツールを呼ばせる**方式を既定にしています（API消費0回）。精度優先で別モデルに投げ直したい場合は `TOKA_EMOTION_MODE=separate` です。

### 記憶の3層化

生の発話ログを1件ずつベクトル化すると、「ユーザー: うん」のような断片が想起の邪魔をします。

* `events` — 生ログ（200件FIFO）
* `episodes` — 8ターンごとに要約した会話の塊。**想起の主対象**
* `profile` — ユーザーについて判明した恒久的事実。毎ターン全量をプロンプトに載せる

想起は距離だけでなく新しさでも重み付けします（「意味は近いが3ヶ月前」より「やや遠いが昨日」を優先）。

### 感情の永続化と、それがLLMに届くこと

感情はJSONに永続化され、起動時に復元されます。加えて重要なのは、`client.chats.create()` をやめて履歴を自前で持つようにしたことです。`chats.create` は生成時の `config` を固定するため、システムプロンプトに埋め込んだ感情値が**起動時のまま永久に更新されません**でした。現在はシステムプロンプトを毎ターン組み直すので、感情の変化が同一セッション中の応答トーンに反映されます。

### 自発発話を「うるさくない」ものにする

放っておくと単に鬱陶しい存在になるので、抑制を三重にかけています。

1. **LLMに拒否権**: 衝動は「今声をかけるべきか」という問いとして渡り、不要なら `SILENT` を返す。大半は黙ります。
2. **クールダウン**: 既定15分。反応がなければ間隔を倍にしていきます。
3. **静音時間帯とキルスイッチ**: 時間帯設定に加え、`set_proactive` ツール経由で「今忙しいから黙ってて」と会話で止められます。

---

## 5. 禁止事項・注意事項

* **APIキー（.env）のコミット厳禁**
  `.env` は絶対にプッシュしないでください（`.gitignore` 済み）。
* **音声ライブラリの利用規約遵守**
  `VOICEVOX` で生成した音声を利用・公開する際は、該当キャラクターの利用規約（クレジット表記の有無など）を必ず確認してください。
* **ツールの安全設計**
  ファイル削除・任意のシェル実行は**そもそも登録していません**。拒否リストではなく「登録しない」で担保しています。アプリ起動は `config.ALLOWED_APPS` の許可リスト方式、ファイル操作は `ALLOWED_FILE_ROOTS` 配下の読み取りのみです。外に影響が出る操作は `risk="confirm"` を付けて音声で確認を取ります。
* **Bluetooth / 無線オーディオ機器に関する注意**
  無線機器はOSのオーディオプロファイル競合（HFP/A2DP切り替え）やサンプリングレート不一致でマイク入力が不安定になります。**有線マイク・有線スピーカー（またはPC内蔵）を推奨**します。全二重動作ではエコー回り込みの観点からもヘッドセットが有利です。
* **動作環境の制限**
  マイクとオーディオI/Oを直接制御するため、ローカル実機での動作が前提です。Docker等のコンテナやクラウドではオーディオデバイスを掴めません。
* **Windows環境のビルドエラー**
  STT が sherpa-onnx（prebuilt wheel）になったため、以前 PyAudio で必要だった `pipwin` 対応は不要になりました。それでも `Microsoft Visual C++ 14.0 or greater is required` が出る場合は、[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/ja/visual-cpp-build-tools/) を「C++ によるデスクトップ開発」にチェックを入れてインストールしてください。

---

## 6. セットアップ

**前提**: Python 3.12.7 / VOICEVOX が起動していること。

```bash
git clone https://github.com/5K13209/AI-Voice-Assistant-
cd AI-Voice-Assistant-

pip install -r requirements.txt

# ONNX モデルを取得（約3GB、初回のみ）
python scripts/fetch_models.py

# APIキーを設定
cp .env-example .env
# .env を開いて GEMINI_API_KEY を入力
```

旧バージョンから移行する場合は、埋め込みモデルが変わっているのでベクトルの入れ直しが必要です。

```bash
python scripts/migrate_memory.py          # 変更内容を確認するだけ
python scripts/migrate_memory.py --apply  # 実際に書き込む
```

### 実行

VOICEVOX を起動した状態で:

```bash
python main.py
# または
python -m toka --help

python -m toka --model pro          # Gemini Pro を使う
python -m toka --stt whisper        # 旧 STT 経路と比較する
python -m toka --no-proactive       # 自発発話を止める
python -m toka --list-devices       # 音声デバイス一覧
python -m toka -v                   # 詳細ログ
```

初回起動時は声紋の登録（5秒×5回）が走ります。

### 主な環境変数

| 変数 | 既定 | 用途 |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | — | 必須 |
| `TOKA_GEMINI_RPM` | `5` | 1分あたりのリクエスト上限。有料枠なら上げる |
| `TOKA_STT_ENGINE` | `sherpa` | `whisper` で旧経路 |
| `TOKA_EMOTION_MODE` | `tool` | `separate` / `keyword` |
| `TOKA_VOICE_THRESHOLD` | `0.45` | 声紋照合の閾値 |
| `TOKA_VOICE_AUTH` | `1` | `0` で照合を無効化（デバッグ用） |
| `TOKA_PROACTIVE` | `1` | `0` で自発発話を無効化 |
| `VOICEVOX_URL` | `http://127.0.0.1:50021` | |
| `VOICEVOX_SPEAKER_ID` | `6` | |

### 調整用のコマンド

```bash
# 声紋の閾値を決める（本人同士のスコア分布を出す）
python -m toka.services.auth

# WAV を1本デコードして、エンジンごとの精度と速度を比べる
python -m toka.services.stt --wav path/to.wav
python -m toka.services.stt --wav path/to.wav --engine whisper
```
