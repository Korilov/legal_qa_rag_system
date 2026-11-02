# retrieval/retriever.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "index")


class LawRetriever:
    def __init__(self, top_k=3):
        self.top_k = top_k
        print("正在加载检索模型...")
        self.embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")

        index_path = os.path.join(INDEX_DIR, "law.faiss")
        docs_path = os.path.join(INDEX_DIR, "docs.json")

        print("正在加载 FAISS 索引...")
        with open(index_path, "rb") as f:
            index_data = f.read()

        index_array = np.frombuffer(index_data, dtype=np.uint8)
        self.index = faiss.deserialize_index(index_array)

        with open(docs_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)
        print("检索器加载完毕！")

    def retrieve(self, query):
        q_vec = self.embedder.encode(query, normalize_embeddings=True)
        q_vec = np.array([q_vec]).astype("float32")
        scores, ids = self.index.search(q_vec, 10)

        results = []
        seen_texts = set()

        # 准备一份常见的地名列表（可以自己扩充）
        city_keywords = ["上海", "北京", "广州", "深圳", "安徽", "长春", "江苏", "浙江"]

        # 判断用户问题中是否包含特定地名
        target_city = None
        for city in city_keywords:
            if city in query:
                target_city = city
                break

        for score, idx in zip(scores[0], ids[0]):
            if idx != -1:
                doc = self.docs[idx].copy()
                if doc['text'] in seen_texts:
                    continue
                seen_texts.add(doc['text'])

                score = float(score)
                law_name = doc['law_name']

                if target_city:
                    if target_city in law_name:
                        score += 1.0
                    elif "中华人民共和国" in law_name:
                        score += 0.2
                    else:
                        score -= 0.5
                else:
                    if "中华人民共和国" in law_name:
                        score += 0.8
                    elif "开发区" in law_name or "园区" in law_name:
                        score -= 0.5
                    elif "省" in law_name or "市" in law_name:
                        score -= 0.2

                doc["retrieval_score"] = score
                results.append(doc)

        results.sort(key=lambda x: x["retrieval_score"], reverse=True)
        return results[:self.top_k]