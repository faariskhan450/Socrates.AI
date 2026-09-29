# Socratic AI: Philosophical RAG Pipeline

## Overview
A Retrieval-Augmented Generation (RAG) pipeline designed to ingest, process, and query foundational philosophical texts. The current infrastructure processes raw `.txt` files (e.g., Plato's Dialogues, Descartes' Meditations), segments them dynamically using recursive chunking, and stores them in a local ChromaDB vector database using OpenAI's `text-embedding-3-small` model.

## Tech Stack
* **Language:** Python
* **LLM & Embeddings:** OpenAI API
* **Framework:** LangChain
* **Vector Database:** ChromaDB

## Installation & Setup
1. Clone this repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file in the root directory and add your OpenAI API key: `OPENAI_API_KEY="your_api_key_here"`
6. Place your `.txt` files into a `docs/` directory.

## Usage
Run the ingestion script to process the documents and build the vector database:
```bash
python ingestion_pipeline.py