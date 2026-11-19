import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv(override=True)
assert os.getenv("GROQ_API_KEY"), "ERROR:GROQ_API_KEY not found in .env file!"

from groq import Groq
active_models = [
    m.id for m in Groq().models.list().data
    if not any(x in m.id.lower() for x in ("whisper", "tts", "guard", "orpheus", "compound"))
]
selected_model = next(
    (m for m in active_models if any(k in m.lower() for k in ("llama", "qwen", "gpt"))),
    active_models[0]
)
print("Using Active Model:", selected_model)
llm = ChatGroq(model=selected_model, temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient(path="./qdrant_db")
vector_store = QdrantVectorStore(
    client=client,
    collection_name="legal_contracts",
    embedding=embeddings,
)

# STEP 3: FlashRank Re-ranker setup
FlashrankRerank.model_rebuild()
reranker = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=4)


def retrieve_and_rerank(query: str, doc_name: Optional[str] = None, top_k: int = 10):
    """Stage 1: Retrieves the top 10 chunks from Qdrant.
       Stage 2: Re-ranks them using FlashRank and returns the top 4 best chunks."""
    qdrant_filter = None
    if doc_name:
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=doc_name),
                )
            ]
        )

    raw_docs = vector_store.similarity_search(query, k=top_k, filter=qdrant_filter)
    if not raw_docs:
        return []

    reranked_docs = reranker.compress_documents(documents=raw_docs, query=query)
    return reranked_docs

class AgentState(TypedDict):
    user_query: str
    task_type: str
    doc_1: Optional[str]
    doc_2: Optional[str]
    answer: str


# --- NODE 1: SUPERVISOR / ROUTER AGENT ---
def router_node(state: AgentState):
    query = state["user_query"]
    doc_1 = state.get("doc_1")
    doc_2 = state.get("doc_2")

    if doc_1 and doc_2 and any(w in query.lower() for w in ["compare", "difference", "vs", "versus"]):
        return {"task_type": "compare"}

    prompt = f"""Classify the user's legal request into exactly ONE category:
- 'qa': asking a specific question about clauses, terms, or facts
- 'summarize': asking for a summary or overview of a contract
- 'compare': asking to compare two documents

User Request: "{query}"
Reply with ONLY one word: qa, summarize, or compare."""

    decision = llm.invoke(prompt).content.strip().lower()
    if "compare" in decision:
        task = "compare"
    elif "summarize" in decision:
        task = "summarize"
    else:
        task = "qa"

    print(f"[Router Agent] Detected Task: {task.upper()}")
    return {"task_type": task}


# --- NODE 2: Q&A AGENT ---
def qa_agent(state: AgentState):
    docs = retrieve_and_rerank(state["user_query"], doc_name=state.get("doc_1"))
    if not docs:
        return {"answer": "No relevant clause related to this question was found in the documents."}

    context = "\n\n".join(
    [f"[{d.metadata.get('source', 'Unknown')}]\n{d.page_content}" for d in docs]
)

    prompt = f"""You are an expert AI Legal Assistant. Answer the user's question strictly using the provided contract excerpts.
Always cite the source document name in brackets like after each point.

Context:
{context}

Question: {state['user_query']}"""

    response = llm.invoke(prompt)
    return {"answer": response.content}


# --- NODE 3: SUMMARIZER AGENT ---
def summarizer_agent(state: AgentState):
    doc_name = state.get("doc_1")
    if not doc_name:
        return {"answer": "Please select a document (doc_1) to summarize."}

    topics = [
        "parties involved effective date purpose of agreement",
        "payment terms fees compensation billing",
        "obligations confidentiality intellectual property",
        "termination notice period liability indemnification governing law"
    ]
    collected_chunks = []
    seen_texts = set()

    for topic in topics:
        docs = retrieve_and_rerank(topic, doc_name=doc_name, top_k=6)
        for d in docs:
            if d.page_content not in seen_texts:
                seen_texts.add(d.page_content)
                collected_chunks.append(d.page_content)

    context = "\n\n---\n\n".join(collected_chunks[:8])

    prompt = f"""You are a Senior Legal Analyst. Create a clear, structured executive summary of the contract '{doc_name}' using the excerpts below.
Include these headings:
1. Parties & Purpose
2. Key Obligations & Scope
3. Payment / Commercial Terms
4. Termination, Liability & Governing Law
5. Potential Risks / Red Flags

Contract Excerpts:
{context}"""

    response = llm.invoke(prompt)
    return {"answer": response.content}


# --- NODE 4: DOCUMENT COMPARISON AGENT ---
def comparison_agent(state: AgentState):
    doc_1 = state.get("doc_1")
    doc_2 = state.get("doc_2")

    if not doc_1 or not doc_2:
        return {"answer": "Selecting two documents (doc_1 and doc_2) is required for comparison."}

    search_focus = f"{state['user_query']} termination liability indemnity confidentiality payment terms"
    docs_1 = retrieve_and_rerank(search_focus, doc_name=doc_1, top_k=8)
    docs_2 = retrieve_and_rerank(search_focus, doc_name=doc_2, top_k=8)

    text_1 = "\n\n".join([d.page_content for d in docs_1])
    text_2 = "\n\n".join([d.page_content for d in docs_2])

    prompt = f"""You are a Legal Contract Comparison Specialist. Compare '{doc_1}' and '{doc_2}' based on the user's request: "{state['user_query']}".

Provide:
1. A Markdown Side-by-Side Comparison Table (Columns: Clause / Aspect | {doc_1} | {doc_2})
2. Key Differences & Risk Analysis (which contract is stricter or more favorable).

--- DOCUMENT 1 ({doc_1}) ---
{text_1}

--- DOCUMENT 2 ({doc_2}) ---
{text_2}"""

    response = llm.invoke(prompt)
    return {"answer": response.content}


# --- BUILD LANGGRAPH WORKFLOW ---
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("qa_agent", qa_agent)
workflow.add_node("summarizer_agent", summarizer_agent)
workflow.add_node("comparison_agent", comparison_agent)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    lambda state: state["task_type"],
    {
        "qa": "qa_agent",
        "summarize": "summarizer_agent",
        "compare": "comparison_agent",
    },
)

workflow.add_edge("qa_agent", END)
workflow.add_edge("summarizer_agent", END)
workflow.add_edge("comparison_agent", END)

legal_graph = workflow.compile()


# --- TEST BLOCK  ---
if __name__ == "__main__":
    try:
        sample_files = [
            f for root, _, files in os.walk("./data") for f in files if f.endswith(".txt")
        ]
        f1, f2 = sample_files[0], sample_files[1]

        print(f"\nTesting with File 1: {f1}")
        print(f"Testing with File 2: {f2}\n" + "=" * 50)

        res_qa = legal_graph.invoke({
            "user_query": "What are the termination conditions and governing law?",
            "doc_1": f1,
            "doc_2": None,
        })
        print("\n--- Q&A AGENT OUTPUT ---\n", res_qa["answer"])

        res_cmp = legal_graph.invoke({
            "user_query": "Compare termination, liability, and confidentiality clauses.",
            "doc_1": f1,
            "doc_2": f2,
        })
        print("\n--- COMPARISON AGENT OUTPUT ---\n", res_cmp["answer"])

    finally:
        client.close()