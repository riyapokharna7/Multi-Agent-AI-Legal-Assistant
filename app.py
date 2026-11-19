import os
import streamlit as st
from graph import legal_graph

st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️", layout="wide")

st.title("⚖️ Multi-Agent AI Legal Assistant")
st.caption("Powered by LangGraph, Qdrant Vector DB & FlashRank Re-ranking")

@st.cache_data
def get_contract_list():
    files = []
    for root, _, filenames in os.walk("./data"):
        for f in filenames:
            if f.endswith(".txt"):
                files.append(f)
    return sorted(files)

contract_files = get_contract_list()

with st.sidebar:
    st.header("📄 Select Contracts")
    st.write(f"Total indexed contracts: **{len(contract_files)}**")

    doc_1_choice = st.selectbox(
        "Document 1 (For Q&A/Summarize):",
        options=["All Documents (No Filter)"] + contract_files,
        index=1 if contract_files else 0
    )

    doc_2_choice = st.selectbox(
        "Document 2 (For Comparison):",
        options=["None"] + contract_files,
        index=2 if len(contract_files) > 1 else 0
    )

    doc_1 = None if doc_1_choice == "All Documents (No Filter)" else doc_1_choice
    doc_2 = None if doc_2_choice == "None" else doc_2_choice

    st.divider()
    st.subheader("⚡ Quick Actions")
    summarize_btn = st.button("📝 Summarize Document 1", use_container_width=True)
    compare_btn = st.button("🔍 Compare Doc 1 vs Doc 2", use_container_width=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Select a contract from the left sidebar to ask any question, generate a summary, or compare two contracts."
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input("Enter your legal question (e.g., What is the termination notice period?)...")

if summarize_btn:
    if doc_1:
        user_prompt = f"Summarize the contract {doc_1} in detail."
    else:
        st.warning("Please select a document (doc_1) to summarize.")

if compare_btn:
    if doc_1 and doc_2:
        user_prompt = f"Compare {doc_1} and {doc_2} on termination, liability, indemnity, and payment terms."
    else:
        st.warning("Please select both documents (doc_1 and doc_2) for comparison.")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agents are searching and re-ranking clauses..."):
            result = legal_graph.invoke({
                "user_query": user_prompt,
                "doc_1": doc_1,
                "doc_2": doc_2,
            })
            agent_used = result.get("task_type", "qa").upper()
            answer = f"**🤖 Active Agent:** `{agent_used}`\n\n{result['answer']}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})