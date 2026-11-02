# eval/evaluate.py
import sys
import os
import json
import re
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.generate import LegalRAGSystem

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FILE = os.path.join(BASE_DIR, "data", "processed", "val.jsonl")  # 用验证集当测试集


def extract_laws(text):
    """提取答案中的法条号"""
    return set(re.findall(r"第[一二三四五六七八九十百千0-9]+条", text))


def evaluate():
    system = LegalRAGSystem()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    # 取前 100 条做测试，避免太慢
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[:100]

    metrics = {"baseline": [], "rag": []}

    print("开始评估...")
    for line in tqdm(lines):
        item = json.loads(line)
        q = item["question"]
        gold_answer = item["answer"]
        gold_laws = extract_laws(gold_answer)

        # 1. 基线
        ans_base = system.generate_pure_seq2seq(q)
        pred_laws_base = extract_laws(ans_base)
        rl_base = scorer.score(gold_answer, ans_base)['rougeL'].fmeasure
        hit_base = len(pred_laws_base & gold_laws) > 0 if gold_laws else False

        # 2. RAG
        ctxs = system.retriever.retrieve(q)
        ans_rag = system.generate_rag(q, ctxs)
        pred_laws_rag = extract_laws(ans_rag)
        rl_rag = scorer.score(gold_answer, ans_rag)['rougeL'].fmeasure
        hit_rag = len(pred_laws_rag & gold_laws) > 0 if gold_laws else False

        metrics["baseline"].append({"rougeL": rl_base, "hit": hit_base})
        metrics["rag"].append({"rougeL": rl_rag, "hit": hit_rag})

    # 汇总
    for mode in ["baseline", "rag"]:
        avg_rl = np.mean([m["rougeL"] for m in metrics[mode]])
        hit_rate = np.mean([m["hit"] for m in metrics[mode]])
        print(f"\n[{mode.upper()}] ROUGE-L: {avg_rl:.4f}, 法条命中率: {hit_rate:.2%}")


if __name__ == "__main__":
    evaluate()