import streamlit as st
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Page Configuration
st.set_page_config(page_title="RAG Document Search & Summarizer", layout="wide")
st.title("📄 Ever Quint: Document Search & Summarization System")
st.markdown("Search through the Wikipedia corpus and generate grounded, length-controlled summaries using LLMs.")

# 2. Load Corpus and Models (Cached for performance)
@st.cache_resource
def initialize_system():
    # Load dataset
    dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
    corpus_texts = dataset["passage"][:5000]
    
    # BM25 index
    tokenized_corpus = [text.lower().split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_embeddings = embedding_model.encode(corpus_texts, convert_to_tensor=True)
    
    # Summarization model
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    return corpus_texts, bm25, embedding_model, corpus_embeddings, tokenizer, model

with st.spinner("Initializing system, loading models and corpus... Please wait."):
    corpus_texts, bm25, embedding_model, corpus_embeddings, tokenizer, model = initialize_system()

# 3. Sidebar Configuration for Bonus Features
st.sidebar.header("⚙️ Configuration")
summary_length = st.sidebar.selectbox(
    "Select Summary Length", 
    options=["short", "medium", "detailed"], 
    index=0
)
top_k = st.sidebar.slider("Number of Documents to Retrieve", min_value=1, max_value=5, value=3)

# Quick Auto-Suggestion Buttons in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Query Suggestions")
example_queries = [
    "Was Abraham Lincoln the sixteenth President of the United States?",
    "When was Monroe elected president?",
    "Tell me about American history."
]
selected_suggestion = st.sidebar.selectbox("Choose a sample query", ["Select a query..."] + example_queries)

# 4. Main Query Input
query_input = st.text_input("Enter your search query:", value="" if selected_suggestion == "Select a query..." else selected_suggestion)

if st.button("Search & Summarize", type="primary"):
    if not query_input.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Searching corpus and generating summary..."):
            # A. Hybrid Search Implementation
            tokenized_query = query_input.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
            
            query_embedding = embedding_model.encode(query_input, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            semantic_ranked_indices = cos_scores.argsort(descending=True)[:top_k * 2].tolist()
            
            # Reciprocal Rank Fusion (RRF)
            rrf_scores = {}
            k_constant = 60
            for rank, idx in enumerate(bm25_ranked_indices):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_constant + (rank + 1))
            for rank, idx in enumerate(semantic_ranked_indices):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_constant + (rank + 1))
                
            final_sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
            
            retrieved_results = []
            for idx in final_sorted_indices[:top_k]:
                retrieved_results.append({
                    "id": idx,
                    "text": corpus_texts[idx],
                    "rrf_score": rrf_scores[idx]
                })
            
            # B. Local Summarization
            context = " ".join([doc['text'] for doc in retrieved_results])
            input_text = f"summarize: {context}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
            
            max_len = 60 if summary_length == "short" else (120 if summary_length == "medium" else 200)
            min_len = 20 if summary_length == "short" else 40
            
            summary_ids = model.generate(inputs["input_ids"], max_length=max_len, min_length=min_len, length_penalty=2.0, num_beams=4, early_stopping=True)
            generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # C. Display Results
            st.subheader("📌 Generated Summary")
            st.success(generated_summary)
            
            st.markdown("---")
            st.subheader("📚 Retrieved Source Documents (Context)")
            for i, doc in enumerate(retrieved_results):
                with st.expander(f"Source {i+1} [Doc ID: {doc['id']} | RRF Score: {doc['rrf_score']:.4f}]"):
                    st.write(doc['text'])