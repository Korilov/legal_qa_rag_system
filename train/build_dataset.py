# train/build_dataset.py
import json
import random
import os

# 动态获取当前项目的根目录 (即 D:\学习\Python中文自然语言处理\legal_qa)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_JSONL = os.path.join(BASE_DIR, "data", "raw", "law_articles.jsonl")
OUT_TRAIN = os.path.join(BASE_DIR, "data", "processed", "train.jsonl")
OUT_VAL = os.path.join(BASE_DIR, "data", "processed", "val.jsonl")

# 确保输出目录存在
os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)

def build_dataset():
    pairs = []
    print("正在读取法条并构建问答对...")

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            law_name = item.get("law_name", "")
            article_no = item.get("article_no", "")
            text = item.get("text", "")

            # 过滤太短或无效的法条
            if len(text) < 15 or article_no == "全文":
                continue

            # 构造多种不同问法（1条法条变5条数据）
            q_templates = [
                f"《{law_name}》{article_no}的内容是什么？",
                f"{law_name}规定{article_no}是什么？",
                f"请告诉我{law_name}的{article_no}。",
                f"根据{law_name}，{article_no}是怎么规定的？",
                f"{article_no}在{law_name}中具体说了什么？"
            ]

            a = f"根据《{law_name}》{article_no}，{text}"

            # 把5个问题都存进去
            for q in q_templates:
                pairs.append({"question": q, "answer": a})

            pairs.append({"question": q, "answer": a})

    # 打乱数据
    random.shuffle(pairs)

    # 划分训练集 (90%) 和验证集 (10%)
    split_idx = int(len(pairs) * 0.9)
    train_data = pairs[:split_idx]
    val_data = pairs[split_idx:]

    # 写入文件
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for p in train_data:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for p in val_data:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"数据构建完成！训练集: {len(train_data)} 条，验证集: {len(val_data)} 条")
    print(f"保存至 {OUT_TRAIN} 和 {OUT_VAL}")


if __name__ == "__main__":
    build_dataset()