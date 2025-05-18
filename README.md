# In-Memory RAG Application

A Python-based **Retrieval-Augmented Generation (RAG)** application that combines document retrieval and generative AI to answer queries about Lebanese hospitals and the medical situation. The application processes CSV, PDF, Word, and Excel files, indexes their content, and uses a generative AI model to provide context-aware answers.

---

## Features

- **Data Ingestion**: Supports CSV, PDF, Word, and Excel files for data extraction and indexing.
- **Indexing**: Generates embeddings for text chunks and saves them for future use.
- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with generative AI to answer user queries.
- **Interactive Query Interface**: Allows users to input queries and receive answers with and without RAG context.
- **File Upload Support**: Users can upload files dynamically, and the application will include their content in the search results.
- **Persistence**: Saves the indexed data to avoid re-indexing on subsequent runs.

---

## Table of Contents

- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Demo](#demo)
- [License](#license)

---

## Setup

### Prerequisites

- Python 3.8 or higher
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### API Configuration

1. Create a `.env` file in the project root.
2. Add your **Google Generative AI API key**:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

### Data Preparation

- Place your CSV, PDF, Word, and Excel files in the `data/` directory.
- Update file paths and relevant columns in `config.py` if needed.

---

## Usage

1. **Run the Application**:
   ```bash
   python main.py
   ```

2. **Query Interface**:
   - Enter your query about Lebanese hospitals or the medical situation.
   - Upload files dynamically to include their content in the search results.
   - Type `quit` to exit the application.

---

## Configuration

The `config.py` file contains the following settings:

- **Data Paths**:
  - `CSV_FILEPATH`: Path to the CSV file.
  - `PDF_FILEPATH`: Path to the PDF file.

- **Model Names**:
  - `EMBEDDING_MODEL_NAME`: Name of the embedding model.
  - `GENERATIVE_MODEL_NAME`: Name of the generative model.

- **Indexing Settings**:
  - `CHUNK_SIZE`: Number of characters per text chunk for PDFs.
  - `CHUNK_OVERLAP`: Overlap between consecutive chunks.

- **Retrieval Settings**:
  - `TOP_K`: Number of top results to retrieve during RAG.

---

## Project Structure

```
RAG Project/
├── assets/                   # Directory for images and demo GIFs
│   ├── process_flow.png      # Image for the "How It Works" section
│   ├── demo.gif              # Demo GIF for the application
├── data/                     # Directory for input data files (CSV, PDF, Word, Excel)
│   ├── hospitals_leb.csv     # Example CSV file
│   ├── WHO_Article.pdf       # Example PDF file
├── src/                      # Source code directory
│   ├── ingest.py             # Handles data ingestion and indexing
│   ├── utils.py              # Utility functions for embeddings, similarity, and file operations
│   ├── rag.py                # Implements the RAG pipeline
├── ui/       
│   ├── app.py                # Gradio Interface 
├── config.py                 # Configuration file for paths, models, and settings
├── main.py                   # Entry point for the application
├── index.json                # Saved index file (generated after indexing)
├── .env                      # Environment file for API keys
└── README.md                 # Documentation file
```

---

## How It Works

1. **Data Ingestion**:
   - The application reads data from CSV, PDF, Word, and Excel files.
   - Text is split into chunks, and embeddings are generated for each chunk.

2. **Indexing**:
   - The embeddings and text chunks are saved to an in-memory index.
   - The index is persisted to a file (`index.json`) to avoid re-indexing on subsequent runs.

3. **Querying**:
   - Users input queries through an interactive interface.
   - The application retrieves relevant context from the index and generates answers using the RAG pipeline.

4. **Output**:
   - The application displays answers with and without RAG context for comparison.

![Process Flow](assets/process_flow.png)

---

## Demo

Below is a demo of the application in action:

![Demo](assets/demo.gif)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
