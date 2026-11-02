# ⚖️ 基于 Seq2Seq 与 RAG 的劳动法律智能问答系统

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)

## 📖 项目简介

本项目为黔南民族师范学院计算机与信息学院的本科设计项目。

针对传统 Seq2Seq 模型在法律问答场景下容易产生“事实幻觉（编造法条）”、“知识更新困难”以及“缺乏可解释性（黑盒）”的三大痛点，本项目构建了一个基于**检索增强生成（RAG）**与**可解释性归因算法**的劳动法律智能问答系统。



系统以真实爬取的国家法律法规数据库为基石，采用微调后的 `T5-base` 作为生成底座，通过 FAISS 向量检索引擎引入外部最新法条知识，并创新性地引入了“删除法（Deletion Attribution）”计算每一条法条的贡献度，实现了生成结果的有据可查、直观可释。



## ✨ 核心亮点与改进点

1. **事实性改进（RAG）**：摒弃纯参数化记忆，通过 BGE 模型将用户问题向量化，从 FAISS 库中检索 Top-K 相关法条作为外部证据，大幅降低模型幻觉。
2. **可解释性改进（归因分析）**：采用“删除法”（Deletion Attribution）计算每条检索法条对最终答案的贡献度，并在前端以进度条形式可视化，打破深度学习“黑盒”。
3. **法理规则干预（Neuro-symbolic AI）**：为解决纯算法“重短文本、轻长文本”导致的法理位阶倒置问题，在检索层与归因层引入了动态惩罚机制，强制系统优先信任“中华人民共和国”级别的上位法。
4. **知识热更新（外置索引）**：法律修订时，无需重新训练整个模型，只需增量更新 FAISS 向量库，即可实现新法条的秒级生效。


## 👨‍💻 团队成员

* **组长**：任聪（负责前端界面与架构设计）
* **组员**：蒋煜华（负责RAG检索调优、核心算法与模型训练）
* **组员**：张世成（负责爬虫开发与数据处理）
* **组员**：舒才美（负责归因模块与系统测试）
* **指导教师**：海涛老师



## 🛠️ 技术栈

* **深度学习框架**：PyTorch
* **模型与依赖**：HuggingFace Transformers (`uer/t5-base-chinese-cluecorpussmall`), Datasets, Accelerate
* **检索与向量库**：Sentence-Transformers (`BAAI/bge-small-zh-v1.5`), FAISS
* **Web 交互框架**：Streamlit
* **数据处理**：Requests, BeautifulSoup4, python-docx



## ✨ 核心功能与改进点

1. **法律知识库自动化构建**：通过爬虫技术抓取国家法律法规数据库，自动提取结构化法条（8000+条），构建本地专业知识库。
2. **事实性改进（RAG）**：摒弃纯参数化记忆，通过 BGE 模型将用户问题向量化，从 FAISS 库中检索 Top-K 相关法条作为外部证据输入解码器，大幅降低模型幻觉。
3. **知识更新改进（外置索引）**：法律修订时，无需重新训练整个模型，只需增量更新 FAISS 向量库，实现知识的秒级热更新。
4. **可解释性改进（归因分析）**：采用“删除法”（Deletion Attribution）计算每条检索法条对最终答案的贡献度，并在前端以进度条形式可视化，让用户清晰知道答案的依据来源。
5. **法理规则干预（Neuro-symbolic AI）**：为解决纯算法“重短文本、轻长文本”导致的法理位阶倒置问题，在检索层与归因层引入了动态惩罚机制，强制系统优先信任“中华人民共和国”级别的上位法。



## 📂 目录结构

```bash
legal_qa/
  ├── app/                # Web 前端界面 (Streamlit)
  │   └── demo.py
  ├── crawler/            # 数据采集与解析
  │   └── crawl_final.py  # 最终版爬虫
  ├── data/               # 数据目录
  │   ├── processed/      # 训练集与验证集 (jsonl)
  │   └── raw/            # 爬取的原始法条 (jsonl)
  ├── eval/               # 自动化评测脚本
  │   └── evaluate.py
  ├── index/              # FAISS 向量索引文件
  ├── models/             # 微调后的生成模型 (T5)
  ├── rag/                # RAG 核心生成与归因逻辑
  │   └── generate.py
  ├── retrieval/          # 检索器构建与查询
  │   ├── build_index.py
  │   └── retriever.py
  ├── train/              # 模型训练与数据构建
  │   ├── build_dataset.py
  │   └── train_seq2seq.py
  └── README.md
```

## ⚠️ 重要：模型权重与数据文件下载

> **由于 GitHub 对单文件大小有限制（100MB），训练好的模型权重（约 800MB）未直接上传。**

请点击下方链接下载模型文通过网盘分享的文件：

📦 **模型权重下载**：[百度网盘链接（提取码：1234）](https://pan.baidu.com/s/1R6OaPDj9GTF8nin3pPNeCw?pwd=1234)

- - **放置路径**：解压后，将 `model.safetensors` 及其他 `.json` 文件放入 `legal_qa/models/seq2seq_legal/` 目录下。

*(注：如果你希望从零开始复现整个训练过程，可以直接运行 `crawler` 和 `train` 目录下的脚本。)**



## 🚀 快速运行指南

### 1. 环境配置

建议使用 Anaconda 创建虚拟环境 (Python 3.10)：

```bash
conda create -n legal_qa python=3.10 -y
conda activate legal_qa
```

安装依赖（请根据你的 CUDA 版本选择安装 PyTorch）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate sentencepiece protobuf tiktoken sentence-transformers faiss-cpu streamlit rouge_score
```

### 2. 启动 Web 系统

确保你的本地已经存在 `models/seq2seq_legal` 和 `index` 文件夹，直接运行：

```bash
streamlit run app/demo.py
```

浏览器将自动打开 `http://localhost:8501`，您可以自由切换“纯 Seq2Seq”与“RAG 增强”模式，体验可解释性归因。



## 📊 实验效果展示

### 1. 自动化评测对比 (测试集 100 条)

| 模型                 | ROUGE-L 分数               | 法条引用命中率       | 平均响应时间 |
| ------------------ | ------------------------ | ------------- | ------ |
| 纯 Seq2Seq 基线       | [填入你的基线分数，如 0.15]        | [如 12.0%]     | 0.8s   |
| **本系统 (RAG+规则干预)** | **[填入你的 RAG 分数，如 0.35]** | **[如 65.0%]** | 1.5s   |

### 2. 可解释性归因演示

https://showcase.png/

> **案例分析**：在检索“劳动合同试用期”问题时，系统准确检索到《中华人民共和国劳动合同法》第十九条，并计算其贡献度高达 94.2%。针对部分地方法规干扰，系统通过法理位阶惩罚机制，强制压制了地方条例的权重。

### 3. （可选）重新训练模型

如果您希望从零开始复现整个过程：

# 1. 爬取数据（生成 law_articles.jsonl）

```bash
python crawler/crawl_final.py
```

# 2. 构建问答数据集

```bash
python train/build_dataset.py
```

# 3. 构建 FAISS 向量索引

```bash
python retrieval/build_index.py
```

# 4. GPU 微调模型

```bash
python train/train_seq2seq.py
```

## ⚠️ 免责声明

**本系统仅供学术实验、教学演示与法律信息参考，不构成正式的法律意见。** 系统的推理结果可能存在偏差，在涉及具体法律纠纷时，请务必咨询具备执业资格的专业律师，并以现行有效的法律法规及当地司法机关的解释为准。

*Copyright © 2026 黔南民族师范学院计算机与信息学院. All Rights Reserved.*
