"""記憶。JSON の逐次ログと Chroma のベクトル検索を束ねる。

旧実装からの主な変更:

* 埋め込みモデルを all-MiniLM-L6-v2（英語専用）から多言語版に替えた。
  日本語の会話を英語モデルで埋め込んでいたため、想起がほぼ機能していなかった。
  次元は同じ 384 なので旧コレクションと混ぜると静かに壊れる。名前を変える。
* Chroma の ID を str(time.time()) から uuid4 に。同一 tick で衝突して
  Chroma が例外を投げていた。
* 記憶を 3 層にした。生ログ(events) / 要約(episodes) / 恒久的事実(profile)。
  想起の主対象は episodes で、profile は毎ターン全量をプロンプトに入れる。
* 想起を距離だけでなく新しさでも重み付けする。
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from typing import Any

from .. import config

log = logging.getLogger(__name__)

# 想起の再ランク付けで、この秒数だけ古いと距離が 2 倍に見積もられる。
# 「意味は近いが 3 ヶ月前の話」より「やや遠いが昨日の話」を優先させたい。
RECENCY_HALF_LIFE = 7 * 24 * 3600


class MemoryManager:
    def __init__(self) -> None:
        self.memory: dict[str, Any] = self._load()
        self._collection = None
        self._embedder = None
        self._turns_since_episode = 0

    # =========================
    # ▼ 永続化 (JSON)
    # =========================

    def _load(self) -> dict[str, Any]:
        empty = {
            "events": [],
            "episodes": [],
            "profile": [],
            "voice_refs": [],
            "toka_emotion": dict(config.DEFAULT_EMOTION),
        }
        try:
            with open(config.MEMORY_FILE, encoding="utf-8") as f:
                loaded = json.load(f)
        except FileNotFoundError:
            return empty
        except (json.JSONDecodeError, OSError) as exc:
            # 旧実装は裸の except で全記憶を無言で初期化していた。
            # 壊れたファイルは退避してから作り直す。
            backup = config.MEMORY_FILE.with_suffix(f".broken-{int(time.time())}.json")
            log.error("memory.json を読めません (%s)。%s に退避します。", exc, backup.name)
            try:
                shutil.copy2(config.MEMORY_FILE, backup)
            except OSError:
                log.exception("退避に失敗")
            return empty

        # 古い memory.json に新しいキーが無くても落ちないようにする。
        return {**empty, **loaded}

    def save(self) -> None:
        config.MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        # 書き込み中に落ちても元ファイルを壊さないよう、一時ファイル経由にする。
        tmp = config.MEMORY_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)
        tmp.replace(config.MEMORY_FILE)

    # =========================
    # ▼ ベクトル DB (遅延初期化)
    # =========================

    def _ensure_vectors(self) -> None:
        """SentenceTransformer のロードは数秒かかるので、初回使用まで遅らせる。"""
        if self._collection is not None:
            return

        import chromadb
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self._collection = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION
        )
        self._embedder = SentenceTransformer(config.EMBED_MODEL)
        log.info(
            "記憶ベクトル: %s / %d 件",
            config.CHROMA_COLLECTION,
            self._collection.count(),
        )

    def _embed(self, text: str) -> list[float]:
        self._ensure_vectors()
        return self._embedder.encode(text).tolist()

    def _index(self, text: str, metadata: dict[str, Any]) -> None:
        self._ensure_vectors()
        self._collection.add(
            documents=[text],
            embeddings=[self._embed(text)],
            # 旧実装は str(time.time())。同一 tick で衝突すると Chroma が投げる。
            ids=[uuid.uuid4().hex],
            metadatas=[metadata],
        )

    # =========================
    # ▼ 書き込み
    # =========================

    def store_event(self, role: str, text: str, emotion: dict[str, int]) -> None:
        """生の発話ログ。JSON にのみ積む（ベクトル化は episode 側で行う）。"""
        self.memory["events"].append(
            {
                "role": role,
                "text": text,
                # 参照のまま持つと全イベントが同じ dict を指し、履歴が
                # 最新の感情値で塗り潰される。必ずコピーを取る。
                "emotion": dict(emotion),
                "time": time.time(),
            }
        )
        if len(self.memory["events"]) > config.MAX_EVENTS:
            self.memory["events"] = self.memory["events"][-config.MAX_EVENTS :]
        self.save()

        if role == "user":
            self._turns_since_episode += 1

    def store_episode(self, summary: str, emotion: dict[str, int]) -> None:
        """会話の塊を要約したもの。想起の主対象。"""
        self.memory["episodes"].append({"text": summary, "time": time.time()})
        self.save()
        self._index(summary, {"kind": "episode", "time": time.time()})
        self._turns_since_episode = 0
        log.info("エピソードを記憶: %s", summary[:60])

    def store_fact(self, fact: str) -> bool:
        """ユーザーについて判明した恒久的な事実。毎ターンプロンプトに入る。

        既知の事実なら False を返す（LLM が同じことを何度も覚えようとする）。
        """
        fact = fact.strip()
        if not fact:
            return False
        if any(fact == existing["text"] for existing in self.memory["profile"]):
            return False

        self.memory["profile"].append({"text": fact, "time": time.time()})
        self.save()
        self._index(fact, {"kind": "profile", "time": time.time()})
        log.info("覚えた: %s", fact)
        return True

    # =========================
    # ▼ 読み出し
    # =========================

    @property
    def voice_refs(self) -> list[str]:
        return self.memory.get("voice_refs", [])

    @voice_refs.setter
    def voice_refs(self, refs: list[str]) -> None:
        self.memory["voice_refs"] = refs
        self.save()

    @property
    def emotion(self) -> dict[str, int]:
        saved = self.memory.get("toka_emotion") or {}
        # 将来キーを増やしても古い memory.json で KeyError にならないようにする。
        return {**config.DEFAULT_EMOTION, **saved}

    @emotion.setter
    def emotion(self, value: dict[str, int]) -> None:
        self.memory["toka_emotion"] = dict(value)
        self.save()

    def profile_text(self) -> str:
        facts = [entry["text"] for entry in self.memory.get("profile", [])]
        return "\n".join(f"- {f}" for f in facts)

    def should_summarize(self) -> bool:
        return self._turns_since_episode >= config.EPISODE_EVERY_TURNS

    def recent_events(self, count: int) -> list[dict[str, Any]]:
        return self.memory["events"][-count:]

    def recall(self, query: str, top_k: int | None = None) -> str:
        """関連する記憶を引く。距離と新しさの両方で並べ替える。"""
        top_k = top_k or config.RECALL_TOP_K
        self._ensure_vectors()

        if self._collection.count() == 0:
            return ""

        # 再ランク付けするので、多めに取ってから絞る。
        fetch = min(top_k * 4, self._collection.count())
        try:
            results = self._collection.query(
                query_embeddings=[self._embed(query)],
                n_results=fetch,
            )
        except Exception:
            log.exception("記憶の検索に失敗")
            return ""

        documents = results.get("documents") or [[]]
        distances = results.get("distances") or [[]]
        metadatas = results.get("metadatas") or [[]]
        if not documents[0]:
            return ""

        now = time.time()
        scored = []
        for doc, dist, meta in zip(
            documents[0], distances[0], metadatas[0] or [{}] * len(documents[0])
        ):
            age = now - float((meta or {}).get("time", now))
            # 古いほど距離を割り増しする。半減期ごとに 1 段階不利になる。
            penalty = 1.0 + max(0.0, age) / RECENCY_HALF_LIFE
            scored.append((dist * penalty, doc))

        scored.sort(key=lambda pair: pair[0])
        return "\n".join(doc for _, doc in scored[:top_k])

    def transcript_for_summary(self) -> str:
        """直近のやり取りを要約用のテキストに整形する。"""
        events = self.recent_events(config.EPISODE_EVERY_TURNS * 2)
        lines = []
        for event in events:
            speaker = "ユーザー" if event.get("role") == "user" else "トーカ"
            lines.append(f"{speaker}: {event['text']}")
        return "\n".join(lines)
