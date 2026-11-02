# rag/generate.py
import os

# 强制加入环境配置，防止分词器报错
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "D:/hf_cache"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BertTokenizer
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.retriever import LawRetriever

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = r"D:\学习\Python中文自然语言处理\legal_qa\models\seq2seq_legal"


class LegalRAGSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"加载模型到 {self.device}...")

        self.tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(self.device)
        self.model.eval()
        print("实际模型所在设备:", next(self.model.parameters()).device)

        print("加载检索器中...")
        self.retriever = LawRetriever(top_k=3)

    def generate_pure_seq2seq(self, question):
        """纯 Seq2Seq 基线生成（无检索）"""
        prompt = "回答问题：" + question
        inputs = self.tokenizer(prompt, max_length=256, truncation=True, return_token_type_ids=False,
                                return_tensors="pt")

        # 获取模型所在的真实设备
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                num_beams=1,  # 关键修改：把 3 改成 1（贪婪搜索）
                no_repeat_ngram_size=4,  # 严格限制重复
                early_stopping=True,  # 尽早结束
                repetition_penalty=1.2
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")

    def generate_rag(self, question, contexts):
        """RAG 生成"""
        prompt = self.build_rag_input(question, contexts)
        # 将 max_length 从 512 降到 384 或者 256，大幅减少显存占用
        inputs = self.tokenizer(prompt, max_length=384, truncation=True, return_token_type_ids=False,
                                return_tensors="pt")

        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,  # 限制生成的最大长度，防止废话连篇
                num_beams=1,
                no_repeat_ngram_size=4,
                early_stopping=True,
                repetition_penalty=1.2
            )
        raw_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")

        # 释放显存缓存
        del inputs, outputs
        torch.cuda.empty_cache()

        # ================= 终极兜底逻辑 =================
        if contexts:
            best_law_name = contexts[0]['law_name']
            best_article = contexts[0]['article_no']
            best_text = contexts[0]['text']

            if best_law_name not in raw_answer or len(raw_answer) < 10:
                return f"根据《{best_law_name}》{best_article}规定：{best_text}"

        if "。" in raw_answer:
            raw_answer = raw_answer.split("。")[0] + "。"

        return raw_answer

    def build_rag_input(self, question, contexts):
        evidence = "\n".join(
            [f"[{i + 1}] 《{c['law_name']}》{c['article_no']}：{c['text']}" for i, c in enumerate(contexts)])

        prompt = (
            f"请直接根据以下法律条文原文回答，不要输出无关内容。\n"
            f"法律条文：\n{evidence}\n\n"
            f"问题：{question}\n"
            f"答案："
        )
        return prompt
    def calculate_attribution(self, question, contexts, answer):
        """可解释性：删除法计算贡献度 + 位阶惩罚"""
        # 防御性判断，如果没检索到或者没生成答案，直接返回空
        if not contexts or not answer:
            return []

        def get_logprob(ctxs):
            prompt = self.build_rag_input(question, ctxs)
            inputs = self.tokenizer(prompt, max_length=384, truncation=True, return_token_type_ids=False,
                                    return_tensors="pt")
            labels = self.tokenizer(answer, return_tensors="pt").input_ids

            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            labels = labels.to(model_device)

            with torch.no_grad():
                out = self.model(**inputs, labels=labels)

            loss = -out.loss.item()

            # 清理显存
            del inputs, labels, out
            torch.cuda.empty_cache()

            return loss

        full_score = get_logprob(contexts)
        contributions = []

        # 1. 基础删除法计算
        for i in range(len(contexts)):
            reduced_ctxs = contexts[:i] + contexts[i + 1:]
            reduced_score = get_logprob(reduced_ctxs)
            drop = full_score - reduced_score
            # 如果 drop 是负数说明反而提升了概率，这里取 0 平滑一下
            contributions.append(max(drop, 0.01))

        # 2. 核心：引入法律位阶惩罚（强行干预）
        for i in range(len(contributions)):
            if i >= len(contexts):
                break

            law_name = contexts[i]['law_name']

            # 国家法律：权重直接翻 5 倍
            if "中华人民共和国" in law_name:
                contributions[i] *= 5.0
            # 低阶法规：权重暴跌
            elif "开发区" in law_name or "园区" in law_name:
                contributions[i] *= 0.05
            # 地方性法规：权重暴跌
            elif "省" in law_name or "市" in law_name:
                contributions[i] *= 0.1

        # 3. 归一化（计算百分比）
        total = sum(contributions)
        if total > 0:
            contributions = [c / total for c in contributions]
        else:
            contributions = [1.0 / len(contexts)] * len(contexts)

        return contributions


if __name__ == "__main__":
    sys = LegalRAGSystem()
    q = "劳动合同的试用期最长可以多久？"

    print("\n--- 纯 Seq2Seq 基线 ---")
    print(sys.generate_pure_seq2seq(q))

    print("\n--- RAG 检索与生成 ---")
    ctxs = sys.retriever.retrieve(q)
    ans_rag = sys.generate_rag(q, ctxs)
    print("答案:", ans_rag)

    print("\n--- 可解释性归因 ---")
    attrs = sys.calculate_attribution(q, ctxs, ans_rag)
    for i, (c, score) in enumerate(zip(ctxs, attrs)):
        print(f"依据 {i + 1}: 《{c['law_name']}》{c['article_no']} (贡献度: {score:.3f})")