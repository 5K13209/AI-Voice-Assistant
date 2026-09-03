"""エントリポイント。

中身は toka/ 以下にある。ここにロジックを書かないこと。
旧版はモジュールレベルに直書きしていたため、import した瞬間にマイクを
掴んでモデルを読み込む副作用があり、部分的なテストができなかった。

    python main.py            通常起動
    python -m toka --help     オプション一覧
"""

from toka.runtime import run

if __name__ == "__main__":
    raise SystemExit(run())
