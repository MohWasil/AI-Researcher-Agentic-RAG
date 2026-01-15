# Agentic RAG Research Assistant Assignment

This project implements an Agentic Retrieval-Augmented Generation (RAG) system designed to function as a specialized Research Assistant. The system ingests documents, retrieves relevant information based on user queries, and generates answers grounded in the provided context, with mechanisms to reduce hallucinations.

## Architecture Overview

The system follows a structured pipeline combining data loading, vector storage, and agentic reasoning:

1.  **Data Ingestion & Preprocessing**: Documents are loaded, parsed, and split into manageable chunks.
2.  **Embedding & Indexing**: Text chunks are converted into vector embeddings and stored in a FAISS index for fast retrieval.
3.  **Agentic Query Processing**: Upon receiving a query, the system retrieves relevant chunks and uses an LLM guided by a system prompt to generate an answer.
4.  **Maker-Checker Loop**: The generated answer undergoes a validation check to ensure it aligns with the retrieved context, mitigating hallucinations.

## Key Components

### 1. Meta System Prompt (`META_SYSTEM_PROMPT`)
A foundational prompt defined in `Cell 1` that instructs the LLM on its role as a research assistant, emphasizing:
*   Answering questions based *solely* on provided document context.
*   Avoiding hallucination or fabrication of information.
*   Staying focused, relevant, and concise.

### 2. Data Pipeline (`Cells 2-6`)
*   **Document Loading (`Cell 3`)**: Functions `load_document_text` and `load_all_documents_from_directory` handle reading `.txt` and `.pdf` files from a designated directory.
*   **Text Chunking (`Cell 5`)**: The `chunk_text` function breaks large documents into smaller, overlapping segments suitable for embedding and retrieval.

### 3. Vector Database (`Cells 7-9`)
*   **Embedding Model**: Uses `all-MiniLM-L6-v2` from `sentence-transformers` to convert text into numerical vectors.
*   **FAISS Index**: An efficient `IndexFlatIP` is created and populated with document chunk embeddings using `populate_faiss_index`, enabling fast similarity searches.

### 4. Retrieval Component (`Cell 10`)
*   **`retrieve_relevant_chunks`**: Takes a user query, embeds it, and searches the FAISS index to find the `top_k` most relevant document chunks.

### 5. LLM Integration (`Cells 11-12`)
*   **Model**: Utilizes `google/flan-t5-base`, a seq2seq model, accessed via Hugging Face Transformers.
*   **Pipeline**: A `text2text-generation` pipeline is created for streamlined answer generation.

### 6. Agentic Orchestration & Maker-Checker (`Cell 13`)
*   **`rag_agent_query`**: The core function that:
    *   **Retrieves** context using `retrieve_relevant_chunks`.
    *   **Formats** the prompt for the LLM using `format_prompt_for_llm`, incorporating the `META_SYSTEM_PROMPT`.
    *   **Generates** an initial answer using the LLM (the **Maker** step).
    *   **Checks** the generated answer against the context using `simple_check_answer` (the **Checker** step). This basic check looks for keyword overlap to identify potential hallucinations.
    *   **Refines** the response by adding a cautionary note if the checker flags potential issues.

### 7. Safety & Workflow (`Cell 14`)
*   **`validate_input`**: Implements basic input validation to detect potentially harmful query patterns (e.g., attempts to override system instructions).
*   **`run_full_rag_workflow`**: Orchestrates the entire process from document loading to answer generation, incorporating input validation and the Maker-Checker loop.

## How It Works

1.  Place research documents (`.txt` or `.pdf`) in the `../data` directory.
2.  Run the notebook cells sequentially.
3.  Call `run_full_rag_workflow("Your Question Here")` with a relevant query.
4.  The system will:
    *   Load and chunk all documents.
    *   Embed chunks and build the FAISS index.
    *   Validate the input query.
    *   Retrieve relevant text chunks based on the query.
    *   Generate an answer using the LLM, guided by the `META_SYSTEM_PROMPT` and the retrieved context.
    *   Check the answer for potential inconsistencies with the context.
    *   Return the final answer along with the retrieved chunks used.