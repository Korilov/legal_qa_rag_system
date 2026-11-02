import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
)

# 强制检查并锁定显卡
if not torch.cuda.is_available():
    raise RuntimeError("没有检测到可用的 CUDA 显卡！")
device = torch.device("cuda") # 在 PyTorch 里，cuda 就是你的 4060

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FILE = os.path.join(BASE_DIR, "data", "processed", "train.jsonl")
VAL_FILE = os.path.join(BASE_DIR, "data", "processed", "val.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "seq2seq_legal")

MODEL_NAME = "uer/t5-base-chinese-cluecorpussmall"
MAX_IN = 256
MAX_OUT = 256

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)

print("加载数据中...")
train_ds = load_jsonl(TRAIN_FILE)
val_ds = load_jsonl(VAL_FILE)

print("加载分词器和模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# 强制将模型加载到 RTX 4060 上，并开启梯度检查点
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.gradient_checkpointing_enable()

def preprocess(ex):
    model_input = "回答问题：" + ex["question"]
    inputs = tokenizer(model_input, max_length=MAX_IN, truncation=True)
    labels = tokenizer(text_target=ex["answer"], max_length=MAX_OUT, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

print("预处理数据中，可能需要几分钟...")
train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names)

# 极度省显存的参数配置（专为 8GB 显存定制）
args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,       # 降到 2
    gradient_accumulation_steps=16,      # 翻倍到 16，等效 batch_size 32
    learning_rate=3e-4,
    num_train_epochs=1,                  # 先跑 1 个 epoch 看看效果
    fp16=True,
    optim="adafactor",                   # 使用 Adafactor，大幅降低显存占用
    eval_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    logging_steps=50,
    report_to="none",
    dataloader_num_workers=0,
)

collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    processing_class=tokenizer,
)

print("================ 开始训练 ================")
trainer.train()

print("保存最终模型...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"训练完成，模型已保存到 {OUTPUT_DIR}")