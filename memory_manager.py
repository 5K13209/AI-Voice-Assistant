import time
import json

#　ベクトルDB
import chromadb
from sentence_transformers import SentenceTransformer

class MemoryManager:
    
    def __init__(self):
        self.MEMORY_FILE = "memory.json"
        self.memory = self.load_memory()
        
        print(self.memory)
    # --- ベクトルDB ---
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="toka_memory")

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_voice_refs(self):
        return self.memory.get("voice_refs",[])
    
    def load_memory(self):
        try:
            with open(self.MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {
                "events": [],
                "voice_refs": []
            }

    def save_memory(self):
        with open(self.MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def store_memory(self,text, emotion):
        # ファイル保存
        self.memory["events"].append({
            "text": text,
            # 参照のまま持つと全イベントが同じdictを指し、履歴が最新の感情値で埋まってしまう
            "emotion": dict(emotion),
            "time": time.time()
        })

        if len(self.memory["events"]) > 200:
            self.memory["events"].pop(0)

        self.save_memory()

        # ベクトルDB保存
        embedding = self.embed_model.encode(text).tolist()

        print("Chroma保存開始")
        print(text)

        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[str(time.time())]
        )

        print("Chroma保存成功")
        print("件数:", self.collection.count())

    def recall_memory(self,query, top_k=3):
        embedding = self.embed_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

        print(results)

        if results["documents"]:
            return "\n".join(results["documents"][0])
        return ""
    
    def build_prompt(self,user_input,emotion):
        recalled = self.recall_memory(user_input)
        print("=== 呼び出し記憶 ===")
        print(recalled)
        
        return f"""
            【過去の関連記憶】
            {recalled}

            【現在の感情】
            好感度:{emotion['like']}
            楽しさ:{emotion['fun']}
            怒り:{emotion['anger']}
            悲しさ:{emotion['sad']}
            信頼:{emotion['trust']}

            ユーザー: {user_input}
            """