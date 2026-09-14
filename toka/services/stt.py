"""音声認識。VAD で発話区間を切り出し、認識器に渡す。

既定のエンジンは sherpa-onnx 上の ReazonSpeech Zipformer (k2-v2)。
faster-whisper と比べた実利は速度もさることながら、無音や環境音に対して
「ご視聴ありがとうございました」「チャンネル登録お願いします」の類を
でっち上げないこと。旧実装が必要としていた ignore_phrases による後段
フィルタが、構造的に不要になる。

ただし ReazonSpeech の精度優位は自分のテスト分割での測定が根拠であり、
実環境の音声では Whisper に負けるという独立系ベンチもある。
TOKA_STT_ENGINE=whisper で旧経路に切り替えて比較できるようにしてある。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Protocol

import numpy as np

from .. import config
from ..bus import EventBus
from ..events import PartialTranscript, SpeechDetected, Utterance
from ..model_files import resolve_all

log = logging.getLogger(__name__)

# Silero VAD が一度に受け取るサンプル数。モデル側の制約なのでここは固定。
VAD_WINDOW = 512


class Transcriber(Protocol):
    """認識器の共通インターフェース。float32 [-1,1] の配列を受けて文字列を返す。"""

    def transcribe(self, samples: np.ndarray) -> str: ...


class SherpaTranscriber:
    """sherpa-onnx + ReazonSpeech Zipformer transducer。CPU/int8。"""

    def __init__(self) -> None:
        import sherpa_onnx

        model_dir = config.SHERPA_ASR_DIR
        if not model_dir.exists():
            raise FileNotFoundError(
                f"ASRモデルが見つかりません: {model_dir}\n"
                "先に `python scripts/fetch_models.py` を実行してください。"
            )

        files = resolve_all(model_dir)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(files["encoder"]),
            decoder=str(files["decoder"]),
            joiner=str(files["joiner"]),
            tokens=str(files["tokens"]),
            num_threads=config.SHERPA_NUM_THREADS,
            sample_rate=config.SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            # ReazonSpeech k2-v2 は日本語を文字単位でモデル化している。
            modeling_unit="cjkchar",
        )
        log.info("ReazonSpeech Zipformer をロードしました (%s)", model_dir.name)

    def transcribe(self, samples: np.ndarray) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(config.SAMPLE_RATE, samples)
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()


class WhisperTranscriber:
    """旧経路。A/B 比較用に残してある。"""

    # Whisper が無音に対して吐きがちな定型句。sherpa 経路では不要。
    HALLUCINATIONS = ("ご視聴", "チャンネル登録", "字幕")

    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            config.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
        )
        log.info("faster-whisper %s をロードしました", config.WHISPER_MODEL_SIZE)

    def transcribe(self, samples: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            samples,
            language="ja",
            vad_filter=True,
            vad_parameters={"min_speech_duration_ms": 500},
            condition_on_previous_text=False,
        )
        text = "".join(s.text for s in segments).strip()
        if any(p in text for p in self.HALLUCINATIONS):
            return ""
        return text


def create_transcriber() -> Transcriber:
    if config.STT_ENGINE == "whisper":
        return WhisperTranscriber()
    return SherpaTranscriber()


class STTService:
    """常時流れてくるフレームを VAD に食わせ、確定した発話をバスへ流す。

    認識そのものは CPU バウンドなので executor で回す。イベントループ上で
    直接デコードすると、その間 TTS の再生チャンク供給が止まって音が切れる。
    """

    def __init__(self, bus: EventBus, transcriber: Transcriber | None = None) -> None:
        import sherpa_onnx

        self.bus = bus
        self._transcriber = transcriber or create_transcriber()

        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = str(config.SHERPA_VAD_MODEL)
        vad_config.silero_vad.threshold = config.VAD_THRESHOLD
        vad_config.silero_vad.min_silence_duration = config.VAD_MIN_SILENCE
        vad_config.silero_vad.min_speech_duration = config.VAD_MIN_SPEECH
        vad_config.silero_vad.max_speech_duration = config.VAD_MAX_SPEECH
        vad_config.sample_rate = config.SAMPLE_RATE
        self._vad = sherpa_onnx.VoiceActivityDetector(
            vad_config, buffer_size_in_seconds=30
        )

        # VAD_WINDOW ちょうどで渡す必要があるので、端数をここに溜める。
        # FRAME_SIZE と VAD_WINDOW を一致させる前提にすると、デバイス側の
        # 都合でブロックサイズが変わったときに静かに壊れる。
        self._pending = np.zeros(0, dtype=np.float32)

        # VAD に渡した生音声を一定時間ぶん保持しておく。切り出された区間の
        # 前後を、ここから足して復元する（VAD_PADDING）。
        # _audio_origin は _audio[0] が全体の何サンプル目かを表す。
        self._audio = np.zeros(0, dtype=np.float32)
        self._audio_origin = 0

        self._speaking = False
        self._last_partial = 0.0
        self._partial_text = ""

    @property
    def transcriber(self) -> Transcriber:
        """認識器。声紋登録の音声を文字起こしするのに外から使う。

        モデルのロードは数秒かかるので、登録用に別インスタンスを作らず
        ここを共有する。
        """
        return self._transcriber

    def set_speaking(self, speaking: bool) -> None:
        """TTS 再生中かどうか。再生中は自分の声を拾いやすいので VAD を鈍らせる。

        本命のエコー対策は話者照合（トーカの合成音声はユーザーの声紋に
        一致しないので落ちる）。これはその前段の粗いフィルタ。
        """
        if self._speaking == speaking:
            return
        self._speaking = speaking
        threshold = (
            config.VAD_THRESHOLD_WHILE_SPEAKING if speaking else config.VAD_THRESHOLD
        )
        try:
            self._vad.config.silero_vad.threshold = threshold
        except (AttributeError, TypeError):  # pragma: no cover - 実装依存
            log.debug("VAD閾値の動的変更が使えないため、話者照合のみで防ぐ")

    async def run(self, frames) -> None:
        """capture.frames() を食い続ける。停止はタスクキャンセルで行う。"""
        loop = asyncio.get_running_loop()
        was_speech = False

        async for frame in frames:
            self._pending = np.concatenate([self._pending, frame])

            while len(self._pending) >= VAD_WINDOW:
                window = self._pending[:VAD_WINDOW]
                self._pending = self._pending[VAD_WINDOW:]
                self._vad.accept_waveform(window)
                self._retain(window)

            is_speech = self._vad.is_speech_detected()
            if is_speech and not was_speech:
                self.bus.publish(SpeechDetected())
                self._partial_text = ""
                self._last_partial = time.monotonic()
            was_speech = is_speech

            if is_speech and config.PARTIAL_INTERVAL > 0:
                await self._maybe_partial(loop)

            while not self._vad.empty():
                # front が返すのは VAD 内部バッファへの参照で、pop() を含む
                # 任意のメソッド呼び出しで無効になる。必ず先にコピーを取る。
                # （コピーせずに pop すると、空配列を認識器に渡して
                #   「Invalid input shape: {0,80}」で落ちる）
                segment = self._vad.front
                samples = np.array(segment.samples, dtype=np.float32)
                start = segment.start
                self._vad.pop()
                if len(samples) < config.SAMPLE_RATE * config.VAD_MIN_SPEECH:
                    continue
                await self._finalize(loop, self._with_padding(start, len(samples)))

    def _retain(self, window: np.ndarray) -> None:
        """生音声のリングバッファを更新する。"""
        self._audio = np.concatenate([self._audio, window])
        limit = int(config.SAMPLE_RATE * config.AUDIO_BUFFER_SECONDS)
        if len(self._audio) > limit:
            dropped = len(self._audio) - limit
            self._audio = self._audio[dropped:]
            self._audio_origin += dropped

    def _with_padding(self, start: int, length: int) -> np.ndarray:
        """VAD が切った区間の前後に生音声を足して返す。

        Silero の境界は詰まっていて、語頭の子音や小声の語尾が欠ける。
        実測では元音声の 58% しか拾えず、「これはテスト文です」が丸ごと
        落ちていた。前後 0.3 秒を足すと 75% まで戻る。
        """
        pad = int(config.SAMPLE_RATE * config.VAD_PADDING)
        if pad <= 0 or len(self._audio) == 0:
            return self._slice(start, start + length)
        return self._slice(start - pad, start + length + pad)

    def _slice(self, begin: int, end: int) -> np.ndarray:
        """絶対サンプル位置でリングバッファを切る。範囲外は詰める。"""
        lo = max(begin - self._audio_origin, 0)
        hi = min(end - self._audio_origin, len(self._audio))
        if hi <= lo:
            return np.zeros(0, dtype=np.float32)
        return self._audio[lo:hi].copy()

    async def _maybe_partial(self, loop) -> None:
        """発話の途中経過を再デコードする。伸びていくバッファを都度読む。"""
        now = time.monotonic()
        if now - self._last_partial < config.PARTIAL_INTERVAL:
            return
        self._last_partial = now

        # current_segment は SpeechSegment を返す。内部バッファへの参照なので
        # 他のメソッドを呼ぶ前にコピーを取る。
        samples = np.array(self._vad.current_segment.samples, dtype=np.float32)
        if len(samples) < config.SAMPLE_RATE * 0.3:
            return

        text = await loop.run_in_executor(None, self._transcriber.transcribe, samples)
        if text and text != self._partial_text:
            self._partial_text = text
            self.bus.publish(PartialTranscript(text=text))

    async def _finalize(self, loop, samples: np.ndarray) -> None:
        duration = len(samples) / config.SAMPLE_RATE
        text = await loop.run_in_executor(None, self._transcriber.transcribe, samples)
        if not text:
            log.debug("空の認識結果 (%.1f秒) を破棄", duration)
            return

        # 話者照合はファイル入力なので、確定した区間だけを書き出す。
        wav_path = self._write_wav(samples)
        log.info("あなた: %s", text)
        self.bus.publish(Utterance(text=text, wav_path=wav_path, duration=duration))

    @staticmethod
    def _write_wav(samples: np.ndarray) -> str:
        """話者照合に渡すファイルを書き出す。

        旧実装は temp_file/user_input.wav という固定パスに毎回上書きして
        いた。全二重にすると発話が近接して確定しうるので、照合が前の音声を
        読む前に次の音声で潰される。1 発話 1 ファイルにして、古いものは
        まとめて掃除する。
        """
        import soundfile as sf

        # 旧実装は makedirs せず、.gitkeep でディレクトリが存在することに
        # 依存していた。クローン直後や掃除後に落ちるので毎回作る。
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        path = config.TEMP_DIR / f"utt-{uuid.uuid4().hex[:12]}.wav"
        sf.write(path, samples, config.SAMPLE_RATE, subtype="PCM_16")
        STTService._prune_wavs()
        return str(path)

    @staticmethod
    def _prune_wavs(keep: int = 20) -> None:
        files = sorted(
            config.TEMP_DIR.glob("utt-*.wav"), key=lambda p: p.stat().st_mtime
        )
        for stale in files[:-keep]:
            try:
                stale.unlink()
            except OSError:
                pass


def _cli() -> int:
    """単体デコード。エンジン比較に使う。

        python -m toka.services.stt --wav temp_file/user_input.wav
    """
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="WAV を 1 本デコードする")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--engine", choices=["sherpa", "whisper"])
    args = parser.parse_args()

    if args.engine:
        config.STT_ENGINE = args.engine

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    samples, rate = sf.read(args.wav, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != config.SAMPLE_RATE:
        # モデルは 16 kHz 固定。リサンプルせずに渡すと、速度だけ出て
        # 中身が完全な別物になる（無言で壊れるので必ず変換する）。
        from scipy.signal import resample_poly

        from math import gcd

        divisor = gcd(config.SAMPLE_RATE, rate)
        samples = resample_poly(
            samples, config.SAMPLE_RATE // divisor, rate // divisor
        ).astype(np.float32)
        print(f"{rate} Hz -> {config.SAMPLE_RATE} Hz にリサンプルしました")
        rate = config.SAMPLE_RATE

    started = time.monotonic()
    transcriber = create_transcriber()
    load_time = time.monotonic() - started

    started = time.monotonic()
    text = transcriber.transcribe(samples)
    decode_time = time.monotonic() - started

    audio_seconds = len(samples) / rate
    print(f"\nエンジン : {config.STT_ENGINE}")
    print(f"音声長   : {audio_seconds:.2f} 秒")
    print(f"ロード   : {load_time:.2f} 秒")
    print(f"デコード : {decode_time:.2f} 秒  (RTF {decode_time / audio_seconds:.3f})")
    print(f"結果     : {text or '(空)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
