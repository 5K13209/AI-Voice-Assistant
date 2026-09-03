"""ファイル関連ツール。読み取り専用。

書き込み・削除・移動は登録しない。LLM に消させて困るものは、確認ゲートを
挟むより最初から手が届かない方がよい。触れるディレクトリも
config.ALLOWED_FILE_ROOTS に限定する。
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import config
from .registry import tool

log = logging.getLogger(__name__)

MAX_RESULTS = 20
MAX_READ_CHARS = 4000

# 読み取りを許可する拡張子。バイナリを読み込んでコンテキストを壊さない。
TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".log", ".py", ".js", ".ts", ".tsx", ".jsx", ".html",
    ".css", ".xml", ".sql", ".sh", ".bat", ".ps1", ".c", ".h", ".cpp",
    ".java", ".go", ".rs", ".rb", ".php",
}


def _resolve_inside_allowed(raw: str) -> Path | None:
    """許可ディレクトリ配下に収まっていれば解決済みパスを返す。

    シンボリックリンクや .. を経由した脱出を防ぐため、resolve() したうえで
    is_relative_to で判定する。文字列の前方一致では抜けられてしまう。
    """
    try:
        path = Path(raw).expanduser().resolve()
    except (OSError, RuntimeError):
        return None

    for root in config.ALLOWED_FILE_ROOTS:
        try:
            if path.is_relative_to(root.resolve()):
                return path
        except (OSError, RuntimeError):
            continue
    return None


@tool(
    risk="safe",
    params={"query": "ファイル名に含まれる文字列"},
)
def search_files(query: str) -> str:
    """ドキュメント・ダウンロード・デスクトップからファイル名で検索する。"""
    query = query.strip().lower()
    if not query:
        return "検索語が空です。"

    hits: list[Path] = []
    for root in config.ALLOWED_FILE_ROOTS:
        if not root.exists():
            continue
        try:
            for path in root.rglob("*"):
                if len(hits) >= MAX_RESULTS:
                    break
                if path.is_file() and query in path.name.lower():
                    hits.append(path)
        except (OSError, PermissionError):
            # アクセスできないサブツリーは黙って飛ばす。
            continue
        if len(hits) >= MAX_RESULTS:
            break

    if not hits:
        return f"「{query}」に一致するファイルは見つかりませんでした。"

    lines = [f"{len(hits)}件見つかりました:"]
    for path in hits:
        size_kb = path.stat().st_size / 1024
        lines.append(f"  {path} ({size_kb:.0f} KB)")
    return "\n".join(lines)


@tool(
    risk="safe",
    params={"path": "読み取るファイルの絶対パス"},
)
def read_text_file(path: str) -> str:
    """テキストファイルの中身を読む。許可ディレクトリ配下のみ。"""
    resolved = _resolve_inside_allowed(path)
    if resolved is None:
        allowed = ", ".join(str(p) for p in config.ALLOWED_FILE_ROOTS)
        return f"そのパスは読み取りを許可されていません。読めるのは: {allowed}"

    if not resolved.is_file():
        return f"ファイルが見つかりません: {resolved}"

    if resolved.suffix.lower() not in TEXT_SUFFIXES:
        return f"テキストファイルとして扱えない拡張子です: {resolved.suffix}"

    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"読み取りに失敗しました: {exc}"

    log.info("ファイル読み取り: %s", resolved)
    if len(text) > MAX_READ_CHARS:
        return f"{resolved.name}（先頭{MAX_READ_CHARS}字）:\n{text[:MAX_READ_CHARS]}"
    return f"{resolved.name}:\n{text}"


@tool(
    risk="safe",
    params={"directory": "一覧するディレクトリの絶対パス"},
)
def list_directory(directory: str) -> str:
    """ディレクトリの中身を一覧する。許可ディレクトリ配下のみ。"""
    resolved = _resolve_inside_allowed(directory)
    if resolved is None:
        allowed = ", ".join(str(p) for p in config.ALLOWED_FILE_ROOTS)
        return f"そのパスは許可されていません。見られるのは: {allowed}"

    if not resolved.is_dir():
        return f"ディレクトリではありません: {resolved}"

    try:
        entries = sorted(resolved.iterdir(), key=lambda p: (p.is_file(), p.name))
    except OSError as exc:
        return f"一覧に失敗しました: {exc}"

    if not entries:
        return f"{resolved} は空です。"

    lines = [f"{resolved}:"]
    for entry in entries[:MAX_RESULTS]:
        lines.append(f"  {'[D] ' if entry.is_dir() else '    '}{entry.name}")
    if len(entries) > MAX_RESULTS:
        lines.append(f"  ...他 {len(entries) - MAX_RESULTS} 件")
    return "\n".join(lines)
