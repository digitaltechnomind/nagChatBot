# RAG Chatbot (Stateless)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot using:
- Streamlit
- Ollama (Llama3)
- FAISS vector store

## Features
- PDF ingestion
- Semantic search
- Stateless QA

## Setup

### 1. Install dependencies
uv pip install -r requirements.txt

### 2. Run Ollama
ollama pull llama3
ollama pull nomic-embed-text

### 3. Start app
streamlit run main.py