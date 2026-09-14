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


# 認識結果の末尾に付きうる記号。判定の前に落とす。
PUNCTUATION = "。．.、,！!？?　 "


def ends_with_stop_word(text: str) -> bool:
    """認識結果の末尾が終了語かどうか。

    **末尾だけ**を見るのが要点。「以上のことを踏まえて」のように文中へ
    現れた場合に切ってしまうと、喋っている途中で録音が終わる。
    認識結果には句読点が付かないこともあるので、記号を落としてから比べる。
    """
    cleaned = peel_polite(text)
    if not cleaned:
        return False
    return any(cleaned.endswith(word) for word in config.VOICE_ENROLL_STOP_WORDS)


def peel_polite(text: str) -> str:
    """末尾の記号と丁寧形を落とす。終了語の判定を揺れに強くする。

    「以上です。」を素の endswith("以上") で見ると一致しない。実測でも
    ここが最初に漏れた。
    """
    cleaned = (text or "").strip().rstrip(PUNCTUATION)
    for tail in config.VOICE_ENROLL_POLITE_TAILS:
        if cleaned.endswith(tail):
            cleaned = cleaned[: -len(tail)].rstrip(PUNCTUATION)
            break
    return cleaned


class _KeyStop:
    """キーが押されたかを、標準入力をブロックせずに調べる。

    input() を別スレッドで待つ形にはしない。押されないまま録音が終わると
    そのスレッドが stdin を掴んだままになり、直後の「これで覚えていい？」
    の答えを横取りしてしまう。Windows では msvcrt で覗くだけにする。

    msvcrt が無い環境（Linux / パイプ経由）ではキーでは止められないので、
    終了語と上限秒数だけで止める。
    """

    def __init__(self) -> None:
        try:
            import msvcrt
        except ImportError:  # pragma: no cover - Windows 以外
            self._msvcrt = None
        else:
            self._msvcrt = msvcrt

    @property
    def available(self) -> bool:
        return self._msvcrt is not None

    def drain(self) -> None:
        """録音前に溜まっている入力を捨てる。前の Enter で即終了しないように。"""
        if self._msvcrt is None:
            return
        while self._msvcrt.kbhit():
            self._msvcrt.getch()

    def pressed(self) -> bool:
        if self._msvcrt is None:
            return False
        hit = False
        while self._msvcrt.kbhit():
            self._msvcrt.getch()
            hit = True
        return hit


class RegisterVoice:
    """声紋サンプルの登録。

    録る内容を「何でもよい」から**自己紹介**に変えてある。声紋を取るために
    どうせ十数秒喋ってもらう必要があるので、その音声をそのまま文字起こしして
    プロフィール（名前・呼ばれ方・普段していること）の初期値にも使う。
    ユーザーから見れば、登録が 1 回で済んで名前まで覚えてくれることになる。
    抽出は ProfileEnroller（services/profile.py）が受け持つ。
    """

    # これを下回るサンプルは録り直す。総時間の 3 割は声が入っていてほしい。
    MIN_VOICED_RATIO = 0.3
    MAX_ATTEMPTS = 3

    # 何を喋ればよいか分からないと黙ってしまい、無音のサンプルが登録される。
    # 本数を増やしたときは末尾の汎用文で埋める。
    PROMPTS = (
        "お名前と、どう呼んでほしいかを教えてください。",
        "普段パソコンで何をしているか、仕事や趣味を教えてください。",
        "好きなものや、覚えておいてほしいことを話してください。",
    )
    GENERIC_PROMPT = "何でも構いません。途切れずに話し続けてください。"

    @classmethod
    def prompt_for(cls, index: int) -> str:
        """index 本目に喋ってもらう内容。"""
        if index < len(cls.PROMPTS):
            return cls.PROMPTS[index]
        return cls.GENERIC_PROMPT

    @staticmethod
    def _countdown() -> None:
        """開始までのカウントダウン。1 行に横並びで出す。

        1 行ずつ改行して出していたが、それだと直前に表示した「何を喋るか」
        の指示がスクロールで押し出されて見えなくなる。行を増やさず、
        数字だけ横に伸ばす。
        """
        print("  ", end="", flush=True)
        for count in (3, 2, 1):
            print(f"{count} ", end="", flush=True)
            time.sleep(1)
        print("-> 録音開始", flush=True)

    @classmethod
    def record_one(cls, output_path, samplerate: int, *, transcriber=None) -> str:
        """終了語かキー入力まで録り続ける。

        固定秒数をやめた理由が 2 つある。言い終わっていないのに切られると
        自己紹介が途中で終わるし、逆に言い終わったのに黙って待たされるのも
        無駄になる。自己紹介の長さは人によって違う。

        止まる条件は 3 つ。「以上」などの終了語、キー入力、上限秒数。
        終了語は下限秒数を超えてから探し始める（「以上」だけ言って
        終わられると声紋が作れない）。
        """
        keys = _KeyStop()
        keys.drain()

        if transcriber is not None and keys.available:
            hint = "「以上」と言うか、何かキーを押すと終わります"
        elif transcriber is not None:
            hint = "「以上」と言うと終わります"
        elif keys.available:
            hint = "何かキーを押すと終わります"
        else:
            hint = f"{config.VOICE_ENROLL_MAX_SECONDS:.0f}秒で自動的に終わります"
        print(f"  （{hint}）", flush=True)

        cls._countdown()

        chunks: list[np.ndarray] = []

        def callback(indata, frames_count, time_info, status) -> None:
            # コールバックは PortAudio のスレッドから呼ばれる。バッファは
            # 再利用されるのでコピーを取る。
            chunks.append(indata.copy())

        reason = "上限"
        elapsed = 0.0
        with sd.InputStream(
            samplerate=samplerate, channels=1, dtype="float32", callback=callback
        ):
            started = time.monotonic()
            last_check = 0.0
            while True:
                time.sleep(0.05)
                elapsed = time.monotonic() - started

                if keys.pressed():
                    reason = "キー"
                    break
                if elapsed >= config.VOICE_ENROLL_MAX_SECONDS:
                    break
                if transcriber is None:
                    continue
                if elapsed < config.VOICE_ENROLL_MIN_SECONDS:
                    continue
                if elapsed - last_check < config.VOICE_ENROLL_CHECK_INTERVAL:
                    continue

                last_check = elapsed
                tail = cls._tail(chunks, samplerate)
                if tail is None:
                    continue
                if ends_with_stop_word(transcriber.transcribe(tail)):
                    reason = "終了語"
                    break

        samples = (
            np.concatenate(chunks).ravel()
            if chunks
            else np.zeros(0, dtype=np.float32)
        )
        sf.write(output_path, samples, samplerate)
        print(f"  録音完了（{elapsed:.0f}秒 / {reason}）", flush=True)
        return str(output_path)

    @staticmethod
    def _tail(chunks: list, samplerate: int):
        """終了語の判定に使う末尾を切り出す。

        毎回全体をデコードする必要はない。終了語は末尾にしか現れない。
        """
        if not chunks:
            return None
        samples = np.concatenate(chunks).ravel()
        want = int(config.VOICE_ENROLL_TAIL_SECONDS * samplerate)
        return samples[-want:] if len(samples) > want else samples

    @classmethod
    def record_checked(
        cls, output_path, samplerate: int, *, transcriber=None
    ) -> str | None:
        """声が十分に入るまで録り直す。"""
        for attempt in range(cls.MAX_ATTEMPTS):
            path = cls.record_one(output_path, samplerate, transcriber=transcriber)
            samples, _ = sf.read(path, dtype="float32", always_2d=False)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)

            ratio = voiced_ratio(samples)
            duration = len(samples) / samplerate
            if ratio >= cls.MIN_VOICED_RATIO and duration >= config.VOICE_ENROLL_MIN_SECONDS:
                print(f"  （声の割合 {ratio * 100:.0f}%）", flush=True)
                return path

            remaining = cls.MAX_ATTEMPTS - attempt - 1
            if duration < config.VOICE_ENROLL_MIN_SECONDS:
                trouble = f"短すぎます（{duration:.0f}秒）。"
            else:
                trouble = f"声がほとんど入っていません（{ratio * 100:.0f}%）。"
            print(
                "  " + trouble
                + (f"録り直します（残り{remaining}回）" if remaining else "先に進みます"),
                flush=True,
            )
        return None

    @classmethod
    def register(cls, transcriber=None) -> list[str]:
        """VOICE_ENROLL_SAMPLES 本を録って、そのパスを返す。

        transcriber を渡すと「以上」で録音を終われる。渡さない場合は
        キー入力と上限秒数だけで止まる。

        パスは相対で返す。絶対パスを memory.json に書くと、リポジトリを
        別の場所へ移したときに全部無効になる。
        """
        config.VOICE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        print(f"\n声紋を登録します（全{config.VOICE_ENROLL_SAMPLES}回）")
        print("話し終わったら「以上」と言うか、何かキーを押してください。")
        print("話した内容から、名前などのプロフィールも一緒に覚えます。\n")

        for index in range(config.VOICE_ENROLL_SAMPLES):
            print(f"[{index + 1}/{config.VOICE_ENROLL_SAMPLES}] {cls.prompt_for(index)}")
            path = cls.record_checked(
                config.VOICE_DATA_DIR / f"voice_{index}.wav",
                config.SAMPLE_RATE,
                transcriber=transcriber,
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
