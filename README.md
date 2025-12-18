# Transformer-Based RAG QA Assistant

This repository contains two Retrieval-Augmented Generation (RAG) implementations that compare Transformer-based models with recurrent neural network baselines.

## 1. RAG Transformer (Mistral) vs Seq2Seq LSTM

**File:** `RAG&LSTM.py`

This implementation uses a Transformer-based RAG pipeline powered by **Mistral** and compares its outputs against a **Seq2Seq LSTM encoder–decoder** model.  
A **Gradio-based GUI** is provided for interactive querying and comparison.

## 2. RAG MiniLM vs GRU

**File:** `RAG&GRU.py`

This implementation uses a lightweight RAG pipeline with **MiniLM embeddings** and compares it against a **GRU-based sequence model**.  
A **Streamlit-based GUI** is used for interaction and visualization.
