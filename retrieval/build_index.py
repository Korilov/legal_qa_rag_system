# retrieval/build_index.py
import json
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# 如果你下载 huggingface 模型太慢，取消下面这行注释以使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 动态获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_JSONL = os.path.join(BASE_DIR, "data", "raw", "law_articles.jsonl")
INDEX_DIR = os.path.join(BASE_DIR, "index")
os.makedirs(INDEX_DIR, exist_ok=True)


def build_faiss_index():
    print("正在加载 BGE 向量模型（首次运行会自动下载，约 90MB）...")
    # 使用轻量级的中文向量模型
    embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    docs = []
    vectors = []

    print("正在读取法条并生成向量，请稍候...")
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            law_name = item.get("law_name", "")
            article_no = item.get("article_no", "")
            text = item.get("text", "")

            # 拼接检索文本，带上法规名能提高检索命中率
            search_text = f"《{law_name}》{article_no}：{text}"

            docs.append({
                "law_name": law_name,
                "article_no": article_no,
                "text": text
            })

            # 向量化（归一化后内积=余弦相似度）
            vec = embedder.encode(search_text, normalize_embeddings=True)
            vectors.append(vec)

    vectors_np = np.array(vectors).astype("float32")

    print(f"正在构建 FAISS 索引，共 {len(vectors_np)} 条...")
    # 使用内积索引
    index = faiss.IndexFlatIP(vectors_np.shape[1])
    index.add(vectors_np)

    # 保存文件
    print("正在保存索引文件...")
    index_path = os.path.join(INDEX_DIR, "law.faiss")

    # 解决 Windows 下 faiss 不支持中文路径的问题
    import shutil
    temp_index_path = "temp_law.faiss"
    faiss.write_index(index, temp_index_path)  # 先存到没有中文的当前目录
    shutil.move(temp_index_path, index_path)  # 然后再移动到目标目录

    with open(os.path.join(INDEX_DIR, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    print(f"索引构建完成！文件已保存在 {INDEX_DIR} 目录下。")


if __name__ == "__main__":
    build_faiss_index()