import streamlit as st
from dotenv import load_dotenv
import torch
import re

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from transformers import AutoTokenizer, AutoModelForCausalLM


st.title("🏦 RBI Regulatory RAG Assistant")

# -----------------------------
# 1. Load PDF
# -----------------------------

PDF_PATH = "documents/rbi_psl_2025.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

st.write(f"Loaded {len(documents)} pages from the PDF.")

# -----------------------------
# 2. Split PDF into chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

st.write(f"Created {len(chunks)} chunks.")
# -----------------------------
# Keyword Search using BM25
# -----------------------------
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

tokenized_chunks = [
    tokenize(doc.page_content)
    for doc in chunks
]

bm25 = BM25Okapi(tokenized_chunks)
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
# -----------------------------
# 3. Create LOCAL embeddings
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# -----------------------------
# 4. Create Chroma vector DB
# -----------------------------

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./rbi_chroma_db_v2"
)

st.success("✅ Local embeddings created!")
st.success("✅ Chroma vector database is ready!")

# -----------------------------
# 5. Load LOCAL LLM
# -----------------------------

@st.cache_resource
def load_llm():

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    return tokenizer, model


tokenizer, model = load_llm()

# -----------------------------
# 6. Ask a question
# -----------------------------

question = st.text_input(
    "Ask a question about the RBI Priority Sector Lending document:"
)

if question:
    
    # -----------------------------
    # Hybrid Retrieval
    # -----------------------------

    # 1. Semantic search
    semantic_results = vector_db.similarity_search(
        question,
        k=10
    )

    # 2. Keyword search using BM25
    query_tokens = tokenize(question)

    keyword_results = bm25.get_top_n(
        query_tokens,
        chunks,
        n=10
    )

    # 3. Combine semantic + keyword results
    combined_results = {}

    # Add semantic results
    for rank, doc in enumerate(semantic_results):
        key = doc.page_content
        combined_results.setdefault(key, 0)
        combined_results[key] += 1 / (rank + 1)

    # Add keyword results
    for rank, doc in enumerate(keyword_results):
        key = doc.page_content
        combined_results.setdefault(key, 0)
        combined_results[key] += 1 / (rank + 1)

    # 4. Sort by combined score
    ranked_contents = sorted(
        combined_results,
        key=combined_results.get,
        reverse=True
    )

    # 5. Select top results
    # -----------------------------
# Rerank the retrieved candidates
# -----------------------------

candidate_docs = []

for content in ranked_contents[:10]:
    for doc in chunks:
        if doc.page_content == content:
            candidate_docs.append(doc)
            break

pairs = [
    (question, doc.page_content)
    for doc in candidate_docs
]

scores = reranker.predict(pairs)

reranked = sorted(
    zip(candidate_docs, scores),
    key=lambda x: x[1],
    reverse=True
)

results = [
    doc for doc, score in reranked[:5]
]

    # 6. Create context
context = "\n\n".join(
    doc.page_content for doc in results
)

    # 7. Display retrieved information
st.subheader("🔎 Retrieved Information")
st.write(context)

    # -----------------------------
    # 7. Generate an answer
    # -----------------------------
# -----------------------------
# Generate Answer
# -----------------------------
context = "\n\n".join(
    doc.page_content for doc in results
)
messages = [
    {
        "role": "system",
        "content": """
You are an RBI regulatory information assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Use all relevant information from the context.
- Do not use outside knowledge.
- Do not invent or guess information.
- If the answer is a list, include all items that are explicitly supported by the context.
- Give a direct answer.
"""
    },
    {
        "role": "user",
        "content": f"""
Question:
{question}

Context:
{context}

Answer the question directly and completely.
"""
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False
    )

generated_tokens = outputs[0][
    inputs["input_ids"].shape[1]:
]

answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

st.subheader("🤖 Answer")
st.write(answer)