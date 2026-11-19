# ⚖️ Multi-Agent AI Legal Assistant

An end-to-end multi-agent Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **Qdrant**, **FlashRank**, **Groq**, and **Streamlit** to analyze, summarize, and compare legal contracts.

## 🚀 Key Features

- **Autonomous Intent Routing:** Uses a LangGraph state machine (`router` node) to classify user queries and dynamically dispatch tasks to specialized agents (`QA`, `SUMMARIZER`, or `COMPARISON`).
- **Precision Contract Q&A:** Retrieves relevant clauses across 644 indexed legal contracts, re-ranks results with **FlashRank** for higher contextual precision, and generates grounded answers with source document citations.
- **Single-Document Summarization:** Extracts key obligations, termination terms, governing law, and critical clauses for any selected contract.
- **Side-by-Side Contract Comparison:** Compares two selected contracts across liability, confidentiality, termination, and risk provisions.
- **Interactive Dark-Mode UI:** Built with Streamlit featuring sidebar contract selectors, one-click quick actions, and active agent indicators.

## 🛠️ Tech Stack & Architecture

- **Agent Orchestration:** LangGraph, LangChain
- **LLM Inference:** Groq API (`ChatGroq`)
- **Vector Database & Embeddings:** Qdrant (local vector store indexing **94,567 chunks** across **644 contracts**), HuggingFace Embeddings
- **Re-Ranking:** FlashRank (`FlashrankRerank`)
- **Evaluation & EDA:** RAGAS (`context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`), Jupyter Notebook
- **Frontend:** Streamlit
- **Benchmark Corpora:** CUAD, ContractNLI, MAUD, PrivacyQA

## 📂 Project Structure

```text
Multi-Agent-AI-Legal-Assistant/
├── app.py                              # Streamlit chat interface and sidebar controls
├── graph.py                            # LangGraph workflow, router, and specialized agents
├── ingest.py                           # Document chunking, metadata tagging, and Qdrant ingestion
├── requirements.txt                    # Python dependencies
├── .streamlit/
│   └── config.toml                     # Custom dark theme configuration
├── data/                               # Legal contract corpus (.txt files)
└── RAG_Legal_Docs_Riya_Pokharna.ipynb  # Exploratory data analysis & RAGAS evaluation notebook
```

## ⚙️ Local Setup & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/riyapokharna7/Multi-Agent-AI-Legal-Assistant.git](https://github.com/riyapokharna7/Multi-Agent-AI-Legal-Assistant.git)
   cd Multi-Agent-AI-Legal-Assistant
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Build the Qdrant vector index (run once):**
   ```bash
   python ingest.py
   ```

6. **Launch the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## 👤 Author

**Riya Pokharna**  
- GitHub: [@riyapokharna7](https://github.com/riyapokharna7)
