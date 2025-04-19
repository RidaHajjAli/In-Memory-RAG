import pandas as pd
from pypdf import PdfReader
from .utils import get_embedding 
from typing import List, Dict, Any
import time
import config 

in_memory_index: List[Dict[str, Any]] = []

def load_and_index_csv() -> None:
    """Loads data from CSV (using config paths/columns), generates embeddings, and adds to the in-memory index."""
    global in_memory_index
    filepath = config.CSV_FILEPATH
    relevant_columns = config.CSV_RELEVANT_COLUMNS
    id_column = config.CSV_ID_COLUMN # Optional ID column for source tracking

    print(f"Loading and indexing CSV: {filepath}...")
    try:
        df = pd.read_csv(filepath)
        # Validate columns
        missing_cols = [col for col in relevant_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {filepath}: {missing_cols}. Available columns: {df.columns.tolist()}")
        if id_column and id_column not in df.columns:
             print(f"Warning: ID column '{id_column}' not found. Using row number for source.")
             id_column = None # Disable ID column usage

        start_time = time.time()
        processed_count = 0
        skipped_count = 0

        for index, row in df.iterrows():
            # Create a text chunk from relevant columns, handling potential NaN values
            chunk_parts = []
            for col in relevant_columns:
                value = row[col]
                if pd.notna(value) and str(value).strip(): # Check if value exists and is not just whitespace
                    chunk_parts.append(f"{col}: {str(value).strip()}")

            text_chunk = ", ".join(chunk_parts)

            if not text_chunk: # Skip if row had no usable data in relevant columns
                # print(f"Skipping empty row {index+1} in CSV (no data in relevant columns).")
                skipped_count += 1
                continue

            embedding = get_embedding(text_chunk)

            # Determine source identifier
            if id_column:
                source_id = row[id_column]
                source_name = f"CSV Row (ID: {source_id})"
            else:
                source_name = f"CSV Row Num: {index+1}"


            if embedding is not None and any(e != 0.0 for e in embedding):
                in_memory_index.append({
                    'text': text_chunk,
                    'embedding': embedding,
                    'source': source_name
                })
                processed_count += 1
            else:
                #print(f"Skipping {source_name} due to embedding error or zero vector.")
                skipped_count += 1

            # Optional: time.sleep(0.1)

        end_time = time.time()
        print(f"CSV indexing complete. Processed: {processed_count}, Skipped: {skipped_count} rows in {end_time - start_time:.2f} seconds.")

    except FileNotFoundError:
        print(f"Error: CSV file not found at {filepath}")
    except Exception as e:
        print(f"Error processing CSV file {filepath}: {e}")

def load_and_index_pdf() -> None:
    """Loads text from PDF (using config path/chunk settings), generates embeddings, and adds to the in-memory index."""
    global in_memory_index
    filepath = config.PDF_FILEPATH
    chunk_size = config.CHUNK_SIZE
    chunk_overlap = config.CHUNK_OVERLAP
    source_base_name = "PDF Document"

    print(f"Loading and indexing PDF: {filepath}...")
    try:
        reader = PdfReader(filepath)
        full_text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Add page number hint for context (optional)
                # full_text += f"[Page {i+1}]\n" + page_text + "\n\n"
                full_text += page_text + "\n" # Simpler version

        if not full_text or full_text.isspace():
             print(f"Warning: No text extracted from PDF {filepath}.")
             return

        # Chunking logic (remains the same, but uses config values)
        chunks = []
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            chunk = full_text[start:end]
            chunks.append(chunk)
            # Ensure start moves forward, even if overlap is large/equal to chunk_size
            next_start = start + chunk_size - chunk_overlap
            if next_start <= start : # Prevent getting stuck
                next_start = start + 1
            start = next_start

        start_time = time.time()
        processed_count = 0
        skipped_count = 0
        for i, chunk in enumerate(chunks):
            clean_chunk = chunk.strip()
            if not clean_chunk:
                # print(f"Skipping empty chunk {i+1} in PDF.")
                skipped_count += 1
                continue

            embedding = get_embedding(clean_chunk)
            source_name = f"{source_base_name} Chunk {i+1}"

            if embedding is not None and any(e != 0.0 for e in embedding):
                 in_memory_index.append({
                     'text': clean_chunk,
                     'embedding': embedding,
                     'source': source_name
                 })
                 processed_count += 1
            else:
                print(f"Skipping {source_name} due to embedding error or zero vector.")
                skipped_count += 1

            # Optional: time.sleep(0.1) # Rate limiting

        end_time = time.time()
        print(f"PDF indexing complete. Processed: {processed_count}, Skipped: {skipped_count} chunks in {end_time - start_time:.2f} seconds.")

    except FileNotFoundError:
        print(f"Error: PDF file not found at {filepath}")
    except Exception as e:
        print(f"Error processing PDF file {filepath}: {e}")


def get_indexed_data() -> List[Dict[str, Any]]:
    """Returns the current in-memory index."""
    global in_memory_index
    if not in_memory_index:
        print("Warning: Index is empty. Call load_and_index functions first.")
    return in_memory_index