# app/demo.py
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.generate import LegalRAGSystem

st.set_page_config(page_title="劳动法律智能问答系统", layout="wide")
st.title("⚖️ 基于 Seq2Seq 与 RAG 的劳动法律智能问答系统")

@st.cache_resource
def load_system():
    return LegalRAGSystem()

sys = load_system()

st.sidebar.header("实验设置")
mode = st.sidebar.radio("选择模式", ["RAG 增强模式 (推荐)", "纯 Seq2Seq 基线模式"])
show_attr = st.sidebar.checkbox("显示可解释性归因分析", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "**项目团队：**\n"
    "- 组长：任聪 (负责Web界面)\n"
    "- 组员：蒋煜华 (负责检索与系统集成)\n"
    "- 组员：张世成 (负责数据处理与训练)\n"
    "- 组员：舒才美 (负责测试)\n\n"
    "**指导教师：** 海涛老师"
)
question = st.text_input("请输入您的法律问题：", "劳动合同的试用期最长可以多久？")

if st.button("提交提问") or question:
    if not question.strip():
        st.warning("请输入问题！")
    else:
        if mode == "纯 Seq2Seq 基线模式":
            with st.spinner("模型生成中..."):
                ans = sys.generate_pure_seq2seq(question)
                st.success("### 模型回答")
                st.write(ans)

        else:
            with st.spinner("检索与生成中..."):
                contexts = sys.retriever.retrieve(question)
                ans = sys.generate_rag(question, contexts)

                st.success("### 模型回答")
                st.write(ans)

                st.markdown("---")
                st.markdown("### 📚 依据法条溯源")

                if show_attr:
                    with st.spinner("计算法条贡献度 (删除法)..."):
                        attrs = sys.calculate_attribution(question, contexts, ans)
                else:
                    attrs = [1.0] * len(contexts)

                for i, (c, score) in enumerate(zip(contexts, attrs)):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.metric(label="贡献度", value=f"{score * 100:.1f}%")
                    with col2:
                        st.info(f"**依据 {i + 1}：《{c['law_name']}》{c['article_no']}**\n\n{c['text']}")

st.markdown("---")
st.caption("⚠️ **免责声明**：本系统仅供教学实验与法律信息参考，不构成正式法律意见。")
st.caption("© 2026 黔南民族师范学院 - 计算机与信息学院人工智能专业 - 项目组 (蒋煜华、任聪、张世成、舒才美) 版权所有")