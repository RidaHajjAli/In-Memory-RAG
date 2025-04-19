import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import config
from config import configure_api
from src.utils import get_generative_model, save_index_to_file, load_index_from_file
from src.ingest import load_and_index_csv, load_and_index_pdf, get_indexed_data
from src.rag import generate_answer, query_rag_pipeline

def main():
    """Main function to set up, index data, and run the comparison query loop."""
    print("Starting RAG Comparison Application...")
    print("-" * 60)

    # 1. Configure API (Must be done first)
    try:
        configure_api()
        get_generative_model()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return
    except Exception as e:
        print(f"Error initializing API or models: {e}")
        return

    print("\n--- Data Indexing ---")
    # Load the index from file if it exists
    in_memory_index = load_index_from_file()

    if not in_memory_index:  # If no saved index, perform indexing
        indexing_done = False
        if os.path.exists(config.CSV_FILEPATH):
            load_and_index_csv()
            indexing_done = True
        else:
            print(f"Warning: CSV file not found at {config.CSV_FILEPATH}. Skipping CSV indexing.")

        if os.path.exists(config.PDF_FILEPATH):
            load_and_index_pdf()
            indexing_done = True
        else:
            print(f"Warning: PDF file not found at {config.PDF_FILEPATH}. Skipping PDF indexing.")

        in_memory_index = get_indexed_data()

        if not indexing_done:
            print("\nError: Neither CSV nor PDF found in the data directory. Cannot proceed.")
            return
        elif not in_memory_index:
            print("\nWarning: No data was successfully indexed, possibly due to file errors or empty files.")

        # Save the index to a file for future use
        save_index_to_file(in_memory_index)

    print(f"\nTotal items indexed: {len(in_memory_index)}")
    print("--- Indexing Complete ---")
    print("-" * 60)

    # 3. Start Comparison Query Loop
    print("\n--- Query Interface ---")
    print("Enter your query about Lebanese hospitals or the medical situation.")
    print("Type 'quit' to exit.")

    while True:
        user_query = input("\nQuery: ")
        if user_query.lower() == 'quit':
            break
        if not user_query.strip():
            print("Please enter a valid query.")
            continue

        print("\n" + "=" * 30)
        print("      BEFORE RAG")
        print("=" * 30)
        direct_answer = generate_answer(query=user_query, context_items=None)
        print("\nAnswer (Directly from LLM):")
        print(direct_answer)
        print("=" * 30)

        print("\n" + "#" * 30)
        print("       AFTER RAG")
        print("#" * 30)
        rag_answer = query_rag_pipeline(user_query, in_memory_index, top_k=config.TOP_K)
        print("\nAnswer (Using RAG):")
        print(rag_answer)
        print("#" * 30)

    print("\nExiting RAG application.")

if __name__ == "__main__":
    main()