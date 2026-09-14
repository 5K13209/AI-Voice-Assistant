"""可変長の声紋登録。終了語の判定とカウントダウンの表示。

固定秒数をやめたので、「いつ止まるか」がロジックになった。誤って早く
止まると自己紹介が途中で切れ、止まらないと延々録り続ける。
"""

from __future__ import annotations

from toka import config
from toka.services.auth import RegisterVoice, _KeyStop, ends_with_stop_word
from toka.services.profile import strip_stop_words


class TestEndsWithStopWord:
    def test_以上で止まる(self):
        assert ends_with_stop_word("よろしくお願いします。以上")

    def test_句点つきでも止まる(self):
        assert ends_with_stop_word("以上です。")

    def test_ひらがなでも止まる(self):
        assert ends_with_stop_word("よろしく。いじょう")

    def test_終わりでも止まる(self):
        assert ends_with_stop_word("説明は終わり")
        assert ends_with_stop_word("これでおわり")

    def test_文中の以上では止まらない(self):
        """「以上のことを踏まえて」で切られると喋っている途中で終わる。"""
        assert not ends_with_stop_word("以上のことを踏まえて話します")

    def test_普通の発話では止まらない(self):
        assert not ends_with_stop_word("私の名前はサジです")

    def test_空文字では止まらない(self):
        assert not ends_with_stop_word("")
        assert not ends_with_stop_word("   ")
        assert not ends_with_stop_word(None)

    def test_記号だけでは止まらない(self):
        assert not ends_with_stop_word("。。。")

    def test_設定の語がすべて効く(self):
        for word in config.VOICE_ENROLL_STOP_WORDS:
            assert ends_with_stop_word(f"話しました{word}"), word


class TestStripStopWords:
    def test_末尾の以上を落とす(self):
        assert strip_stop_words("名前はサジです。以上") == "名前はサジです"

    def test_句点つきでも落とす(self):
        assert strip_stop_words("名前はサジです。以上です。") == "名前はサジです"

    def test_重なった終了語も落とす(self):
        assert strip_stop_words("名前はサジです。終わり。以上") == "名前はサジです"

    def test_文中の以上は残す(self):
        text = "以上のことを踏まえて話します"
        assert strip_stop_words(text) == text

    def test_終了語が無ければそのまま(self):
        assert strip_stop_words("名前はサジです") == "名前はサジです"

    def test_終了語だけなら空になる(self):
        assert strip_stop_words("以上") == ""

    def test_空文字は空(self):
        assert strip_stop_words("") == ""
        assert strip_stop_words(None) == ""


class TestPrompts:
    def test_本数ぶんのお題がある(self):
        """お題が無いと何を喋ればよいか分からず、無音が登録される。"""
        for i in range(config.VOICE_ENROLL_SAMPLES):
            assert RegisterVoice.prompt_for(i)

    def test_1本目は名前を訊く(self):
        assert "名前" in RegisterVoice.prompt_for(0)

    def test_お題を超えたら汎用文で埋める(self):
        assert RegisterVoice.prompt_for(999) == RegisterVoice.GENERIC_PROMPT


class TestCountdownLayout:
    def test_改行せず横一列に出す(self, capsys, monkeypatch):
        """1 行ずつ出すと、直前の「何を喋るか」がスクロールで見えなくなる。"""
        monkeypatch.setattr("time.sleep", lambda s: None)
        RegisterVoice._countdown()
        out = capsys.readouterr().out
        # 数字が 1 行に収まっていること。
        first = out.split("\n")[0]
        assert "3" in first and "2" in first and "1" in first
        # 数字ごとに改行していないこと。
        assert out.count("\n") == 1


class TestKeyStop:
    def test_msvcrt_が無くても壊れない(self, monkeypatch):
        """Linux やパイプ経由ではキーで止められない。落ちてはいけない。"""
        keys = _KeyStop()
        keys._msvcrt = None
        assert keys.available is False
        assert keys.pressed() is False
        keys.drain()

    def test_押されていなければ_False(self):
        keys = _KeyStop()

        class _Quiet:
            def kbhit(self):
                return False

            def getch(self):
                raise AssertionError("kbhit が False なら読んではいけない")

        keys._msvcrt = _Quiet()
        assert keys.pressed() is False

    def test_溜まった入力をまとめて捨てる(self):
        keys = _KeyStop()

        class _Buffered:
            def __init__(self):
                self.left = 3
                self.reads = 0

            def kbhit(self):
                return self.left > 0

            def getch(self):
                self.left -= 1
                self.reads += 1
                return b"x"

        fake = _Buffered()
        keys._msvcrt = fake
        assert keys.pressed() is True
        # 押されたことを 1 回検出したら、溜まった分は全部捨てること。
        assert fake.reads == 3
        assert keys.pressed() is False


class _FakeStream:
    """sd.InputStream の代役。with に入ったらフレームを流し込む。"""

    def __init__(self, chunks, callback):
        self._chunks = chunks
        self._callback = callback

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_fake_audio(monkeypatch, tmp_path, transcripts, *, has_key=False):
    """録音ループを、実マイク無しで回せるようにする。

    time.monotonic を自前で進めることで、実時間を待たずに上限や
    チェック間隔の分岐を通せる。
    """
    import numpy as np

    from toka.services import auth as auth_mod

    clock = {"t": 0.0}
    monkeypatch.setattr(auth_mod.time, "monotonic", lambda: clock["t"])
    # sleep のたびに時計を進める。
    monkeypatch.setattr(auth_mod.time, "sleep",
                        lambda s: clock.__setitem__("t", clock["t"] + 0.25))

    state = {"frames": 0}

    def fake_stream(*, samplerate, channels, dtype, callback):
        # 0.25 秒ぶんずつ、呼ばれるたびに積む代わりに、
        # 入場時に上限ぶんまとめて用意しておく。
        block = np.full((samplerate // 4, 1), 0.2, dtype=np.float32)

        class _S:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

        # ループが回る間、毎回 callback が呼ばれる想定にはできないので、
        # 十分な量を先に積む。
        for _ in range(400):
            callback(block, len(block), None, None)
            state["frames"] += 1
        return _S()

    monkeypatch.setattr(auth_mod.sd, "InputStream", fake_stream)

    class _Keys:
        available = has_key

        def drain(self):
            pass

        def pressed(self):
            return has_key

    monkeypatch.setattr(auth_mod, "_KeyStop", lambda: _Keys())

    class _Tr:
        def __init__(self):
            self.calls = 0

        def transcribe(self, samples):
            self.calls += 1
            index = min(self.calls - 1, len(transcripts) - 1)
            return transcripts[index]

    return _Tr()


class TestRecordOneStopping:
    def test_終了語で止まる(self, monkeypatch, tmp_path):
        tr = _install_fake_audio(
            monkeypatch, tmp_path, ["名前はサジです", "以上"]
        )
        out = tmp_path / "v.wav"
        RegisterVoice.record_one(out, config.SAMPLE_RATE, transcriber=tr)
        assert out.exists()
        # 上限まで回らずに止まっていること。
        assert tr.calls <= 3

    def test_終了語が来なければ上限まで回る(self, monkeypatch, tmp_path):
        tr = _install_fake_audio(monkeypatch, tmp_path, ["まだ話しています"])
        out = tmp_path / "v.wav"
        RegisterVoice.record_one(out, config.SAMPLE_RATE, transcriber=tr)
        # 上限 60 秒 / チェック間隔 1.2 秒 ぶんは呼ばれる。
        assert tr.calls > 10

    def test_キーで止まる(self, monkeypatch, tmp_path):
        tr = _install_fake_audio(
            monkeypatch, tmp_path, ["まだ話しています"], has_key=True
        )
        out = tmp_path / "v.wav"
        RegisterVoice.record_one(out, config.SAMPLE_RATE, transcriber=tr)
        # キーが最優先。認識は走らない。
        assert tr.calls == 0

    def test_下限より前は終了語を探さない(self, monkeypatch, tmp_path):
        """「以上」だけ言って終わられると声紋が作れない。"""
        tr = _install_fake_audio(monkeypatch, tmp_path, ["以上"])
        out = tmp_path / "v.wav"
        RegisterVoice.record_one(out, config.SAMPLE_RATE, transcriber=tr)
        import soundfile as sf

        samples, rate = sf.read(out, dtype="float32", always_2d=False)
        assert len(samples) / rate >= config.VOICE_ENROLL_MIN_SECONDS

    def test_認識器が無ければ上限で止まる(self, monkeypatch, tmp_path):
        _install_fake_audio(monkeypatch, tmp_path, [""])
        out = tmp_path / "v.wav"
        RegisterVoice.record_one(out, config.SAMPLE_RATE, transcriber=None)
        assert out.exists()
