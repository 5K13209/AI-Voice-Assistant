"""話者照合。ウェイクワードの代わりに声紋でゲートする。

全二重化した今、この層はもうひとつ役目を持つ。スピーカーから出た自分の声を
マイクが拾い戻しても、VOICEVOX の合成音声はユーザーの声紋 (ECAPA x-vector)
に一致しないので、ここで落ちる。事実上のエコーキャンセラとして働く。

なお声紋の単独運用はリプレイ攻撃に弱い。README が断っているとおり、
これはコンセプト実装として単要素で割り切っている。
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from .. import config
from ..bus import EventBus
from ..events import SpeakerRejected, UserMessage, Utterance

log = logging.getLogger(__name__)


class RegisterVoice:
    """声紋サンプルの登録。"""

    @staticmethod
    def record_one(output_path, duration: int, samplerate: int) -> str:
        # 何の予告もなく録り始めると話し出しに間に合わず、質の低い
        # プロファイルが登録されてしまう。カウントダウンを入れる。
        for count in (3, 2, 1):
            print(f"  {count}...", flush=True)
            time.sleep(1)
        print(f"  録音開始（{duration}秒間、話し続けてください）", flush=True)

        frames = int(duration * samplerate)
        recording = sd.rec(frames, samplerate=samplerate, channels=1, dtype="float32")
        sd.wait()
        sf.write(output_path, recording, samplerate)
        print("  録音完了", flush=True)
        return str(output_path)

    @classmethod
    def register(cls) -> list[str]:
        """VOICE_ENROLL_SAMPLES 本を録って、そのパスを返す。"""
        config.VOICE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        print(f"\n声紋を登録します（全{config.VOICE_ENROLL_SAMPLES}回）")
        for index in range(config.VOICE_ENROLL_SAMPLES):
            print(f"\n[{index + 1}/{config.VOICE_ENROLL_SAMPLES}] 準備してください")
            path = cls.record_one(
                config.VOICE_DATA_DIR / f"voice_{index}.wav",
                config.VOICE_ENROLL_DURATION,
                config.SAMPLE_RATE,
            )
            paths.append(path)

        print("\n声紋登録が完了しました\n")
        return paths


class VoiceAuth:
    """SpeechBrain ECAPA-TDNN によるコサイン類似度照合。"""

    def __init__(self, refs: list[str]) -> None:
        from speechbrain.inference import SpeakerRecognition

        self._model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(config.MODELS_DIR / "spkrec-ecapa-voxceleb"),
        )
        self.refs = refs
        log.info("話者照合モデルをロードしました（参照 %d 件）", len(refs))

    def verify(self, wav_path: str) -> tuple[bool, float]:
        """(通過したか, 最良スコア) を返す。"""
        if not self.refs:
            log.warning("登録済みの声紋がありません")
            return False, 0.0

        scores = []
        for ref in self.refs:
            try:
                score, _ = self._model.verify_files(ref, wav_path)
                scores.append(float(score))
            except Exception:
                log.exception("照合に失敗: %s", ref)

        if not scores:
            return False, 0.0

        # 旧実装はこれを「平均スコア」と表示していたが、実際は最大値。
        best = max(scores)
        return best > config.VOICE_THRESHOLD, best


class AuthService:
    """Utterance を受けて、通れば UserMessage に、落ちれば SpeakerRejected に。"""

    def __init__(self, bus: EventBus, auth: VoiceAuth | None) -> None:
        self.bus = bus
        self._auth = auth

    async def run(self) -> None:
        loop = asyncio.get_running_loop()

        async for event in self.bus.stream(Utterance):
            if not config.VOICE_AUTH_ENABLED or self._auth is None:
                self.bus.publish(UserMessage(text=event.text, score=1.0))
                continue

            ok, score = await loop.run_in_executor(
                None, self._auth.verify, event.wav_path
            )
            if ok:
                log.debug("照合通過 (スコア %.3f)", score)
                self.bus.publish(UserMessage(text=event.text, score=score))
            else:
                log.info("照合で棄却 (スコア %.3f): %s", score, event.text)
                self.bus.publish(SpeakerRejected(text=event.text, score=score))


def calibrate(refs: list[str]) -> None:
    """閾値決めの補助。登録済みサンプル同士のスコア分布を出す。

        python -m toka.services.auth
    """
    auth = VoiceAuth(refs)
    scores = []
    for i, a in enumerate(refs):
        for b in refs[i + 1 :]:
            score, _ = auth._model.verify_files(a, b)
            scores.append(float(score))

    if not scores:
        print("比較できるサンプルがありません")
        return

    array = np.array(scores)
    print(f"\n登録サンプル同士のスコア（本人 vs 本人）: n={len(array)}")
    print(f"  最小 {array.min():.3f} / 平均 {array.mean():.3f} / 最大 {array.max():.3f}")
    print(f"\n現在の閾値: {config.VOICE_THRESHOLD}")
    print("本人同士の最小値より少し下を閾値にするのが目安です。")
    print("他人の声でも試して、そちらが閾値を下回ることを確認してください。")


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        saved = json.loads(config.MEMORY_FILE.read_text(encoding="utf-8"))
        calibrate(saved.get("voice_refs", []))
    except FileNotFoundError:
        print(f"{config.MEMORY_FILE} がありません。先に本体を起動して登録してください。")
