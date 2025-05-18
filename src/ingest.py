import pandas as pd
from pypdf import PdfReader # Assuming you're using pypdf, can be PyPDF2 if that's your lib
from .utils import get_embedding, save_index_to_file  # Add save_index_to_file import
from typing import List, Dict, Any
import time
import config
import re
from docx import Document
import openpyxl
from transformers import AutoTokenizer  # Import tokenizer for token counting

# Initialize tokenizer globally for efficiency
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Replace with your preferred model

in_memory_index: List[Dict[str, Any]] = []

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving important punctuation."""
    # Regex to split by sentence-ending punctuation followed by space and capital, or newline and capital.
    # Handles cases like "Mr. Jones" better by not splitting on all periods.
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(])|(?<=[.!?])\n+(?=[A-Z0-9"\'\(])', text)
    return [s.strip() for s in sentences if s and s.strip()]

def count_tokens(text: str) -> int:
    """Counts the number of tokens in a given text using a transformer tokenizer."""
    try:
        # Add special handling for empty or whitespace text
        if not text or text.isspace():
            return 0
        # Encode the text and count tokens
        tokens = tokenizer.encode(text, truncation=False)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        
        return len(text)

def create_semantic_chunks(text: str, max_tokens: int = 1000, max_chars: int = 2048) -> List[str]:
    """
    Splits text into chunks where each chunk is approximately `max_tokens` tokens.
    Uses sentence boundaries to create more natural chunks.
    """
    if not text or text.isspace():
        return []

    # Split into sentences first
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        # If a single sentence is longer than max_tokens, split it further
        if sentence_tokens > max_tokens:
            # If we have accumulated content, save it first
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0
            
            # Split long sentence into smaller parts
            words = sentence.split()
            temp_chunk = []
            temp_token_count = 0
            
            for word in words:
                word_tokens = count_tokens(word)
                if temp_token_count + word_tokens > max_tokens:
                    if temp_chunk:
                        chunks.append(" ".join(temp_chunk))
                    temp_chunk = [word]
                    temp_token_count = word_tokens
                else:
                    temp_chunk.append(word)
                    temp_token_count += word_tokens
            
            if temp_chunk:
                chunks.append(" ".join(temp_chunk))
            continue

        # If adding this sentence would exceed the token limit, start a new chunk
        if current_token_count + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_token_count = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_token_count += sentence_tokens

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Verify chunk sizes and log any issues
    for i, chunk in enumerate(chunks):
        chunk_tokens = count_tokens(chunk)
        if chunk_tokens > max_tokens:
            print(f"Warning: Chunk {i+1} exceeds token limit ({chunk_tokens} tokens)")
        elif chunk_tokens == 0:
            print(f"Warning: Chunk {i+1} is empty")

    return chunks

def load_and_index_csv() -> None:
    """Loads data from CSV, generates embeddings, and adds to the in-memory index."""
    global in_memory_index
    filepath = config.CSV_FILEPATH

    print(f"Loading and indexing CSV: {filepath}...")
    try:
        df = pd.read_csv(filepath)
        
        id_column = getattr(config, 'CSV_ID_COLUMN', None) # Use getattr for safer access
        columns = df.columns.tolist()
        id_present = False
        if id_column and id_column in columns:
            columns.remove(id_column)
            id_present = True
        else:
            print(f"No CSV_ID_COLUMN configured or found in CSV. Using row numbers for source identification.")

        start_time = time.time()
        processed_count = 0
        skipped_count = 0

        for index, row in df.iterrows():
            chunk_parts = []
            for col in columns:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    chunk_parts.append(f"{col}: {str(value).strip()}")

            text_chunk = ", ".join(chunk_parts)

            if not text_chunk:
                # print(f"Skipping empty row {index+1} in CSV (no data in relevant columns).")
                skipped_count += 1
                continue

            embedding = get_embedding(text_chunk)

            if id_present:
                source_id = row[id_column]
                source_name = f"CSV Row (ID: {source_id})"
            else:
                source_name = f"CSV Row Num: {index+1}"

            if embedding and any(e != 0.0 for e in embedding):
                in_memory_index.append({
                    'text': text_chunk,
                    'embedding': embedding,
                    'source': source_name
                })
                processed_count += 1
            else:
                skipped_count += 1
                print(f"Skipping {source_name} from CSV due to embedding error or zero vector.")


        end_time = time.time()
        print(f"CSV indexing complete. Processed: {processed_count}, Skipped: {skipped_count} rows in {end_time - start_time:.2f}s.")
        
        # Save the updated index to file
        save_index_to_file(in_memory_index)

    except FileNotFoundError:
        print(f"Error: CSV file not found at {filepath}")
    except Exception as e:
        print(f"Error processing CSV file {filepath}: {e}")


def load_and_index_pdf() -> None:
    """Loads text from PDF using semantic chunking, generates embeddings, and adds to the in-memory index."""
    global in_memory_index
    filepath = config.PDF_FILEPATH
    source_base_name = "WHO_Article.pdf"

    print(f"Loading and indexing PDF: {filepath}...")
    try:
        reader = PdfReader(filepath)
        print(f"PDF loaded successfully. Number of pages: {len(reader.pages)}")
        
        full_text_parts = []
        total_pages_processed = 0
        
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if not page_text or page_text.isspace():
                    print(f"Warning: No text extracted from PDF page {i+1}.")
                    continue
                
                print(f"Successfully extracted text from page {i+1}. Length: {len(page_text)}")
                
                # Clean the text
                cleaned_page_text = re.sub(r'\bPage\s*\d+\s*\b', '', page_text, flags=re.IGNORECASE)
                cleaned_page_text = re.sub(r'\r\n', '\n', cleaned_page_text)
                cleaned_page_text = re.sub(r'(?<=\S)\s*\n\s*(?=\S\S)', '\n\n', cleaned_page_text)
                cleaned_page_text = re.sub(r'\n{3,}', '\n\n', cleaned_page_text)
                
                # Add page marker and cleaned text
                full_text_parts.append(f"[Page {i+1}]\n{cleaned_page_text.strip()}")
                total_pages_processed += 1
                print(f"Processed page {i+1} successfully")
            except Exception as e:
                print(f"Error processing page {i+1}: {str(e)}")
                continue

        if not full_text_parts:
            print(f"Error: No text could be extracted from any pages in {filepath}")
            return

        full_text = "\n\n".join(full_text_parts)
        print(f"Total text extracted: {len(full_text)} characters from {total_pages_processed} pages")

        # Create chunks with strict token limit
        chunks = create_semantic_chunks(full_text, max_tokens=1000)
        print(f"Created {len(chunks)} chunks from the PDF text")

        if not chunks:
            print("Error: No chunks could be created from the extracted text")
            return

        start_time = time.time()
        processed_count = 0
        skipped_count = 0

        for i, chunk in enumerate(chunks):
            try:
                clean_chunk = chunk.strip()
                if not clean_chunk:
                    print(f"Skipping empty chunk {i+1}")
                    skipped_count += 1
                    continue

                # Verify token count
                chunk_tokens = count_tokens(clean_chunk)
                if chunk_tokens > 1000:
                    print(f"Warning: Chunk {i+1} exceeds token limit ({chunk_tokens} tokens)")
                    # Truncate the chunk if necessary
                    clean_chunk = clean_chunk[:2048]  # Fallback to character limit

                embedding = get_embedding(clean_chunk)
                if embedding is None:
                    print(f"Failed to generate embedding for chunk {i+1}")
                    skipped_count += 1
                    continue

                if not any(e != 0.0 for e in embedding):
                    print(f"Generated zero vector embedding for chunk {i+1}")
                    skipped_count += 1
                    continue

                source_name = f"{source_base_name} Chunk {i+1}"
                in_memory_index.append({
                    'text': clean_chunk,
                    'embedding': embedding,
                    'source': source_name
                })
                processed_count += 1
                print(f"Successfully processed chunk {i+1} ({chunk_tokens} tokens)")
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")
                skipped_count += 1
                continue

        end_time = time.time()
        print(f"PDF indexing complete. Processed: {processed_count}, Skipped: {skipped_count} chunks in {end_time - start_time:.2f} seconds.")
        
        if processed_count > 0:
            save_index_to_file(in_memory_index)
            print(f"Successfully indexed {processed_count} chunks from the PDF")
        else:
            print("Warning: No chunks were successfully processed from the PDF")

    except FileNotFoundError:
        print(f"Error: PDF file not found at {filepath}")
    except Exception as e:
        print(f"Error processing PDF file {filepath}: {str(e)}")


def get_indexed_data() -> List[Dict[str, Any]]:
    """Returns the current in-memory index."""
    global in_memory_index
    if not in_memory_index:
        print("Warning: Index is empty. Call load_and_index functions first or ensure index file is loaded.")
    return in_memory_index
