"""旧形式の記憶を新形式へ移行する。

旧版からの変更点のうち、データに影響するのは 2 つ。

1. 埋め込みモデルが all-MiniLM-L6-v2（英語専用）から多言語版に変わった。
   次元はどちらも 384 なので、同じコレクションに混ぜても Chroma は
   エラーを出さない。だが英語モデルのベクトルと多言語モデルのベクトルは
   別空間なので、検索結果が静かに壊れる。新コレクションを作って入れ直す。

2. events が {"text": "ユーザー: ..."} という接頭辞つきの 1 本の文字列から、
   {"role": "user", "text": "..."} に変わった。

voice_refs が空で voice_data/ に音声が残っている場合は、そこから復元する。

    python scripts/migrate_memory.py            # 変更内容を表示するだけ
    python scripts/migrate_memory.py --apply    # 実際に書き込む
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from toka import config  # noqa: E402

OLD_COLLECTION = "toka_memory"

# 旧 events のテキストに付いていた話者の接頭辞。
PREFIXES = {"ユーザー: ": "user", "AI: ": "assistant"}


def split_role(text: str) -> tuple[str, str]:
    for prefix, role in PREFIXES.items():
        if text.startswith(prefix):
            return role, text[len(prefix) :]
    return "user", text


def load_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        print(f"警告: {path} を読めません ({exc})。空として扱います。")
        return {}


def migrate_events(memory: dict) -> tuple[list, int]:
    events = memory.get("events", [])
    migrated, changed = [], 0
    for event in events:
        if "role" in event:
            migrated.append(event)
            continue
        role, text = split_role(event.get("text", ""))
        migrated.append(
            {
                "role": role,
                "text": text,
                "emotion": event.get("emotion", dict(config.DEFAULT_EMOTION)),
                "time": event.get("time", time.time()),
            }
        )
        changed += 1
    return migrated, changed


def recover_voice_refs(memory: dict) -> list[str]:
    refs = [r for r in memory.get("voice_refs", []) if Path(r).exists()]
    if refs:
        return refs

    found = sorted(str(p) for p in config.VOICE_DATA_DIR.glob("voice_*.wav"))
    if found:
        print(f"  voice_refs が空なので voice_data/ から {len(found)} 件復元します")
    return found


def reindex(apply: bool) -> int:
    """旧コレクションの文書を、新しい埋め込みモデルで入れ直す。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    existing = {getattr(c, "name", c) for c in client.list_collections()}

    if OLD_COLLECTION not in existing:
        print(f"  旧コレクション {OLD_COLLECTION} はありません。再インデックス不要。")
        return 0

    old = client.get_collection(OLD_COLLECTION)
    data = old.get()
    documents = data.get("documents") or []
    if not documents:
        print(f"  {OLD_COLLECTION} は空です。")
        return 0

    print(f"  {OLD_COLLECTION} から {len(documents)} 件を再インデックスします")
    for doc in documents[:5]:
        print(f"    - {doc[:60]}")
    if len(documents) > 5:
        print(f"    ... 他 {len(documents) - 5} 件")

    if not apply:
        return len(documents)

    from sentence_transformers import SentenceTransformer

    print(f"  埋め込みモデルを読み込み中: {config.EMBED_MODEL}")
    embedder = SentenceTransformer(config.EMBED_MODEL)
    target = client.get_or_create_collection(config.CHROMA_COLLECTION)

    # 既にある内容と重複させないため、同じ本文が入っていれば飛ばす。
    already = set((target.get().get("documents") or []))

    added = 0
    for doc, meta in zip(documents, data.get("metadatas") or [{}] * len(documents)):
        role, text = split_role(doc)
        if text in already:
            continue
        target.add(
            documents=[text],
            embeddings=[embedder.encode(text).tolist()],
            ids=[uuid.uuid4().hex],
            metadatas=[
                {
                    "kind": "episode",
                    "role": role,
                    "time": float((meta or {}).get("time", time.time())),
                }
            ],
        )
        added += 1

    print(f"  {config.CHROMA_COLLECTION} に {added} 件追加しました")
    return added


def drop_test_collections(apply: bool) -> None:
    """開発中に作った検証用コレクションを掃除する。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    for collection in client.list_collections():
        name = getattr(collection, "name", collection)
        if name.startswith("test_"):
            print(f"  検証用コレクション {name} を削除します")
            if apply:
                client.delete_collection(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="実際に書き込む")
    args = parser.parse_args()

    print(f"memory.json : {config.MEMORY_FILE}")
    print(f"chroma      : {config.CHROMA_DIR}")
    print(f"新コレクション: {config.CHROMA_COLLECTION}")
    print(f"埋め込み     : {config.EMBED_MODEL}")
    print()

    memory = load_json(config.MEMORY_FILE)

    print("[1] events の形式変換")
    events, changed = migrate_events(memory)
    print(f"  {len(events)} 件中 {changed} 件を role つきに変換")

    print("[2] voice_refs の確認")
    refs = recover_voice_refs(memory)
    print(f"  {len(refs)} 件")

    print("[3] ベクトルの再インデックス")
    count = reindex(args.apply)

    print("[4] 検証用データの掃除")
    drop_test_collections(args.apply)

    updated = {
        "events": events,
        "episodes": memory.get("episodes", []),
        "profile": memory.get("profile", []),
        "voice_refs": refs,
        "toka_emotion": {
            **config.DEFAULT_EMOTION,
            **(memory.get("toka_emotion") or {}),
        },
    }

    if not args.apply:
        print("\n--- 変更後の memory.json（プレビュー） ---")
        preview = dict(updated)
        preview["events"] = f"<{len(events)} 件>"
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        print("\n書き込むには --apply を付けて実行してください。")
        return 0

    if config.MEMORY_FILE.exists():
        backup = config.MEMORY_FILE.with_suffix(f".bak-{int(time.time())}.json")
        shutil.copy2(config.MEMORY_FILE, backup)
        print(f"\n元の memory.json を {backup.name} に退避しました")

    config.MEMORY_FILE.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{config.MEMORY_FILE} を更新しました（ベクトル {count} 件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
