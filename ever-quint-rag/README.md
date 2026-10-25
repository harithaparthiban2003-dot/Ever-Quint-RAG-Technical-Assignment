\# 📄 Ever Quint RAG Assignment: Document Search \& Summarization



!\[Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge\&logo=python\&logoColor=white)

!\[Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge\&logo=streamlit\&logoColor=white)

!\[Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-yellow?style=for-the-badge\&logo=huggingface\&logoColor=white)

!\[License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)



A professional-grade Retrieval-Augmented Generation (RAG) technical codebase built for the Ever Quint Gen AI Engineer assignment. This system implements hybrid search (lexical + dense semantic ranking via Reciprocal Rank Fusion), local LLM summarization, and an interactive Streamlit user interface.



\---



\## 🏗️ Architecture \& Tech Stack



\* \*\*Data Ingestion:\*\* Hugging Face `datasets` (`rag-mini-wikipedia` / `text-corpus`)\[cite: 1].

\* \*\*Hybrid Retrieval:\*\* 

&#x20; \* \*\*Lexical Search:\*\* `rank-bm25` (BM25Okapi)

&#x20; \* \*\*Semantic Search:\*\* `sentence-transformers` (`all-MiniLM-L6-v2`)

&#x20; \* \*\*Fusion:\*\* Reciprocal Rank Fusion (RRF) algorithm.

\* \*\*Summarization Engine:\*\* Local Hugging Face sequence-to-sequence transformer (`facebook/bart-large-cnn`).

\* \*\*Evaluation Pipeline:\*\* Hit Rate @ K and `rouge-score` (ROUGE-1, ROUGE-2, ROUGE-L).

\* \*\*User Interface:\*\* Streamlit web application.



\---



\## 📁 Project Structure



```text

ever-quint-rag/

│

├── app.py              # Interactive Streamlit web application UI

├── requirements.txt    # Python package dependencies

└── README.md           # Project documentation and setup guide

