import streamlit as st
from app.inference import infer
st.title("📚 Class IX History RAG Assistant")
st.markdown("**Ask questions from *India and the Contemporary World-I* and get precise, concise, context-grounded answers powered by SmolLM2-135M-Instruct.**")
st.markdown("**💡 Try the sample questions in the sidebar to get started.**")
with st.expander("ℹ️ About this project"):
    st.write("""
Class IX History RAG Assistant is a Retrieval-Augmented Generation (RAG) chatbot built around the NCERT Class IX History textbook, India and the Contemporary World-I.

The system first converts the user's question into a semantic embedding using a Sentence Transformer model. FAISS then retrieves the most relevant passages from the textbook. The retrieved context is passed directly to SmolLM2-135M-Instruct, a lightweight instruction-tuned language model, which generates the final answer.

Pipeline:

Question → Embedding → FAISS Retrieval → SmolLM2-135M-Instruct → Answer

The project demonstrates how semantic retrieval and a lightweight language model can be combined to build a domain-specific AI assistant while keeping the system computationally efficient enough for CPU-based deployment.

The assistant is designed to provide concise, context-grounded, and exam-friendly answers based on the provided textbook content. Since the underlying language model is a compact 135M-parameter model, occasional factual or contextual errors may occur. Answers should be verified against the textbook when accuracy is critical.

Tip: Ask clear and specific questions for more precise answers. The assistant is designed for concise responses rather than long-form essays.
    """)
if "messages" not in st.session_state:
    st.session_state.messages=[]
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])
user_input=st.chat_input("Ask a Question...")
if user_input:
    q=user_input.lower().strip()
    if q.startswith(("who is","who was","what is","what was","define")):
        instruction="- Answer in one sentence."
    else:
        instruction="- Answer briefly using only the provided context."
    question_for_model=user_input.strip()+instruction
    st.session_state.messages.append({
        "role":"user",
        "content":user_input
    })
    answer=infer(question_for_model)
    st.session_state.messages.append({
        "role":"assistant",
        "content":answer
    })
    st.rerun()
with st.sidebar:
    st.header("📖 About the Assistant")
    st.markdown("**Source**")
    st.markdown("""
    NCERT Class IX History
    *India and the Contemporary World-I*
    """)
    st.code("""
🔎 Retrieve
Find relevant textbook passages

🤖 Generate
SmolLM2-135M-Instruct generates the answer
    """)
    st.markdown("""
    **Built with**
    FAISS · Sentence Transformers · SmolLM2-135M-Instruct · Streamlit
    """)
    st.sidebar.markdown("### 📚 Example Questions")

    st.sidebar.markdown("""
    - What is reign of terror?
    - Who was Napoleon?
    - Tell me about Karl Marx.
    - What was allied powers?
    - Who was Subhash Chandra Bose?
    - Define communism.
    """)
    if st.button("Clear Chat"):
        st.session_state.messages=[]
        st.rerun()
