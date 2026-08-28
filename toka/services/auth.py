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
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from .. import config
from ..bus import EventBus
from ..events import SpeakerRejected, UserMessage, Utterance

log = logging.getLogger(__name__)


def resolve_ref(path: str) -> Path:
    """voice_refs のパスを絶対パスに直す。

    memory.json には相対パスで持つ（リポジトリを移動しても壊れないように）。
    旧バージョンが書いた絶対パスもそのまま受け付ける。
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else config.ROOT / candidate


def voiced_ratio(samples: np.ndarray, frame: int = 1600) -> float:
    """発話らしいエネルギーを持つフレームの割合。

    録音ボタンだけ押して喋っていない、という登録事故を検出するために使う。
    実際にそれが起きていた: 登録済み 5 本のうち 1 本は完全な無音、残りも
    5 秒中 24〜52% しか声が入っておらず、本人同士のスコアが 0.06〜0.54 と
    使い物にならない範囲に散っていた。
    """
    usable = len(samples) // frame * frame
    if usable == 0:
        return 0.0
    frames = samples[:usable].reshape(-1, frame)
    return float((np.sqrt((frames**2).mean(axis=1)) > 0.01).mean())


class RegisterVoice:
    """声紋サンプルの登録。"""

    # これを下回るサンプルは録り直す。5 秒中 1.5 秒は声が入っていてほしい。
    MIN_VOICED_RATIO = 0.3
    MAX_ATTEMPTS = 3

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
    def record_checked(cls, output_path, duration: int, samplerate: int) -> str | None:
        """声が十分に入るまで録り直す。"""
        for attempt in range(cls.MAX_ATTEMPTS):
            path = cls.record_one(output_path, duration, samplerate)
            samples, _ = sf.read(path, dtype="float32", always_2d=False)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)

            ratio = voiced_ratio(samples)
            if ratio >= cls.MIN_VOICED_RATIO:
                print(f"  （声の割合 {ratio * 100:.0f}%）", flush=True)
                return path

            remaining = cls.MAX_ATTEMPTS - attempt - 1
            print(
                f"  声がほとんど入っていません（{ratio * 100:.0f}%）。"
                + (f"録り直します（残り{remaining}回）" if remaining else "先に進みます"),
                flush=True,
            )
        return None

    @classmethod
    def register(cls) -> list[str]:
        """VOICE_ENROLL_SAMPLES 本を録って、そのパスを返す。

        パスは相対で返す。絶対パスを memory.json に書くと、リポジトリを
        別の場所へ移したときに全部無効になる。
        """
        config.VOICE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        print(f"\n声紋を登録します（全{config.VOICE_ENROLL_SAMPLES}回）")
        print(f"1回につき{config.VOICE_ENROLL_DURATION}秒、途切れずに話し続けてください。")
        print("読み上げる内容は何でも構いません。\n")

        for index in range(config.VOICE_ENROLL_SAMPLES):
            print(f"[{index + 1}/{config.VOICE_ENROLL_SAMPLES}] 準備してください")
            path = cls.record_checked(
                config.VOICE_DATA_DIR / f"voice_{index}.wav",
                config.VOICE_ENROLL_DURATION,
                config.SAMPLE_RATE,
            )
            if path:
                paths.append(str(Path(path).relative_to(config.ROOT)))
            print()

        if not paths:
            print("使えるサンプルが1本も録れませんでした。マイクを確認してください。\n")
        else:
            print(f"声紋登録が完了しました（{len(paths)}本）\n")
        return paths


class VoiceAuth:
    """SpeechBrain ECAPA-TDNN によるコサイン類似度照合。

    verify_files は使わない。SpeechBrain 側がパスをフェッチ対象として扱い、
    絶対パスを渡すとリポジトリのルートに連結して壊れるため（
    'C:\\repo\\C:\\repo\\voice_data\\voice_0.wav' になる）。音声の読み込みは
    こちらで行い、埋め込み同士を直接比べる。

    登録音声の埋め込みは起動時に一度だけ計算して持つ。旧実装は毎発話ごとに
    5 本のファイルを読み直して 5 回forwardしていた。
    """

    def __init__(self, refs: list[str]) -> None:
        from speechbrain.inference import SpeakerRecognition

        kwargs = {
            "source": "speechbrain/spkrec-ecapa-voxceleb",
            "savedir": str(config.MODELS_DIR / "spkrec-ecapa-voxceleb"),
        }
        # SpeechBrain の既定はキャッシュからのシンボリックリンク。Windows では
        # 開発者モードか管理者権限が無いと WinError 1314 で落ちる。コピーに
        # 切り替える（数十MBなので実害はない）。
        try:
            from speechbrain.utils.fetching import LocalStrategy

            kwargs["local_strategy"] = LocalStrategy.COPY
        except ImportError:  # pragma: no cover - 古い speechbrain
            pass

        self._model = SpeakerRecognition.from_hparams(**kwargs)
        self.refs = refs
        self._embeddings = [
            e
            for e in (self._embed(r, check_quality=True) for r in refs)
            if e is not None
        ]
        log.info(
            "話者照合モデルをロードしました（参照 %d/%d 件）",
            len(self._embeddings),
            len(refs),
        )
        self._warn_if_incoherent()

    def _warn_if_incoherent(self) -> None:
        """登録サンプル同士が似ていなければ、閾値以前に登録が失敗している。"""
        if len(self._embeddings) < 2:
            return
        import torch

        scores = [
            float(torch.dot(a, b))
            for i, a in enumerate(self._embeddings)
            for b in self._embeddings[i + 1 :]
        ]
        worst = min(scores)
        if worst < config.VOICE_THRESHOLD:
            log.warning(
                "登録サンプル同士の最小スコアが %.3f で、閾値 %.2f を下回っています。"
                "本人の声でも弾かれる可能性が高いです。voice_data を消して"
                "登録し直すか、TOKA_VOICE_THRESHOLD を下げてください。"
                "（python -m toka.services.auth で分布を確認できます）",
                worst,
                config.VOICE_THRESHOLD,
            )

    @staticmethod
    def _read(path: str) -> np.ndarray | None:
        """16kHz モノラルの float32 として読む。"""
        try:
            data, rate = sf.read(resolve_ref(path), dtype="float32", always_2d=False)
        except Exception:
            log.exception("音声を読めません: %s", path)
            return None

        if data.ndim > 1:
            data = data.mean(axis=1)
        if rate != config.SAMPLE_RATE:
            from math import gcd

            from scipy.signal import resample_poly

            divisor = gcd(config.SAMPLE_RATE, rate)
            data = resample_poly(
                data, config.SAMPLE_RATE // divisor, rate // divisor
            ).astype(np.float32)
        return data

    def _embed(self, path: str, check_quality: bool = False):
        samples = self._read(path)
        if samples is None or len(samples) < config.SAMPLE_RATE * 0.2:
            return None

        if check_quality:
            # 無音の参照を持つと「無音なら通る」抜け穴になる。スコアは
            # 全参照の最大値を取るので、1本でも無音が混ざると環境音で
            # ゲートが開きうる。読み込み時点で外す。
            ratio = voiced_ratio(samples)
            if ratio < RegisterVoice.MIN_VOICED_RATIO:
                log.warning(
                    "%s は声がほとんど入っていないため参照から除外します（%.0f%%）",
                    path,
                    ratio * 100,
                )
                return None

        import torch

        with torch.no_grad():
            vector = self._model.encode_batch(torch.from_numpy(samples).unsqueeze(0))
        return torch.nn.functional.normalize(vector.squeeze(), dim=-1)

    def verify(self, wav_path: str) -> tuple[bool, float]:
        """(通過したか, 最良スコア) を返す。"""
        if not self._embeddings:
            log.warning("登録済みの声紋がありません")
            return False, 0.0

        target = self._embed(wav_path)
        if target is None:
            return False, 0.0

        import torch

        # 旧実装はこれを「平均スコア」と表示していたが、実際は最大値。
        best = max(float(torch.dot(target, ref)) for ref in self._embeddings)
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


def calibrate(refs: list[str], others: list[str] | None = None) -> None:
    """閾値決めの補助。登録済みサンプル同士のスコア分布を出す。

        python -m toka.services.auth
        python -m toka.services.auth 他人の声.wav   # 他人との比較も見る
    """
    import torch

    auth = VoiceAuth(refs)
    embeddings = auth._embeddings
    scores = [
        float(torch.dot(a, b))
        for i, a in enumerate(embeddings)
        for b in embeddings[i + 1 :]
    ]

    if not scores:
        print("比較できるサンプルがありません")
        return

    array = np.array(scores)
    print(f"\n登録サンプル同士のスコア（本人 vs 本人）: n={len(array)}")
    print(f"  最小 {array.min():.3f} / 平均 {array.mean():.3f} / 最大 {array.max():.3f}")

    for path in others or []:
        ok, score = auth.verify(path)
        verdict = "通過（閾値が緩すぎます）" if ok else "棄却"
        print(f"\n{path}: スコア {score:.3f} -> {verdict}")

    print(f"\n現在の閾値: {config.VOICE_THRESHOLD}")
    print("本人同士の最小値より少し下を閾値にするのが目安です。")
    print("他人の声でも試して、そちらが閾値を下回ることを確認してください。")
    print("環境変数 TOKA_VOICE_THRESHOLD で変更できます。")


if __name__ == "__main__":
    import json
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        saved = json.loads(config.MEMORY_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"{config.MEMORY_FILE} がありません。先に本体を起動して登録してください。")
    else:
        calibrate(saved.get("voice_refs", []), sys.argv[1:])
