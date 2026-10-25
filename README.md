# 📄 Ever Quint RAG Technical Assignment: Hybrid Search & Summarization

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-yellow?style=for-the-badge&logo=huggingface&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Welcome to my official submission for the **Ever Quint Generative AI Engineer Technical Assignment**. 

This repository contains a production-ready Retrieval-Augmented Generation (RAG) system built from scratch. It features a hybrid retrieval engine combining lexical keyword search and dense semantic vector search, dynamic length-controlled local summarization using transformers, a rigorous evaluation pipeline, and an interactive web interface.

---

## 🌟 Key Features & Highlights

* **Hybrid Search Engine:** Combines **BM25** (lexical keyword matching) and **MiniLM** (dense semantic vector embeddings) using **Reciprocal Rank Fusion (RRF)** to retrieve high-precision contexts.
* **Local Summarization Engine:** Uses Hugging Face's `facebook/bart-large-cnn` sequence-to-sequence model running entirely locally, with configurable output lengths (`short`, `medium`, `detailed`).
* **Interactive UI:** Built with Streamlit for real-time document search, parameter adjustments, and full source document score inspection.
* **Performance Evaluation:** Validated on standard metrics with **100% Hit Rate @ 3** on test retrieval queries and benchmarked ROUGE scores.

---

## 📁 Repository Structure

```text
ever-quint-rag/
│
├── app.py                  # Main Streamlit web app, Hybrid RAG & BART summarization engine
├── ever-quint-rag.pdf      # Detailed technical report (pre-processing, architecture, metrics & solutions)
├── requirements.txt        # Production dependency environment specification
└── README.md               # Project documentation & execution guide

🛠️ Tech StackLanguage: Python 3.10+Framework / Web UI: StreamlitRetrieval Engine: rank-bm25, sentence-transformers (all-MiniLM-L6-v2)Summarization: transformers (facebook/bart-large-cnn), torchData Processing & Metrics: datasets, rouge-score

🚀 Setup & Local Execution GuideFollow these steps to run the project on your local machine:1. PrerequisitesMake sure you have Python 3.10 or higher installed.
2. InstallationClone this repository and open your terminal inside the project directory:Bashgit clone [https://github.com/](https://github.com/)<your-username>/ever-quint-rag.git cd ever-quint-rag. Install all required Python dependencies:Bashpip install -r requirements.txt
3. Launch the ApplicationRun the Streamlit web application:Bashpython -m streamlit run app.py
The application will automatically launch in your browser at http://localhost:8501.

📊 Benchmark Evaluation ResultsMetricTargetResultTechnical InterpretationHit Rate @ 3Retrieval Quality100.00%Relevant ground-truth document successfully retrieved in top 3 results.ROUGE-1 F1Unigram Overlap0.4501High keyword precision and factual coverage in summaries.ROUGE-2 F1Bigram Overlap0.2648Reliable multi-word phrase matching.ROUGE-L F1Sentence Structure0.3977Strong grammatical coherence and structural similarity to reference texts.

