import os
from src.ingest import load_and_index_csv, load_and_index_pdf, in_memory_index  # Import in_memory_index here
from src.utils import save_index_to_file, load_index_from_file, INDEX_FILE
from ui.app import create_ui

def regenerate_index():
    global in_memory_index
    in_memory_index.clear()  # Clear the existing index in-place
    print("Cleared the in-memory index.")

    # Regenerate embeddings
    print("\n=== Processing CSV Data ===")
    load_and_index_csv()
    csv_count = len(in_memory_index)
    print(f"CSV processing complete. Added {csv_count} items to index.")

    print("\n=== Processing PDF Data ===")
    load_and_index_pdf()
    total_count = len(in_memory_index)
    pdf_count = total_count - csv_count
    print(f"PDF processing complete. Added {pdf_count} items to index.")

    # Save the updated index
    if in_memory_index:  # Ensure index is not empty before saving
        save_index_to_file(in_memory_index)
        print(f"\nIndex regenerated and saved. Total items: {total_count}")
        print(f"Breakdown: {csv_count} CSV items, {pdf_count} PDF items")
        # Reload from file to ensure consistency
        in_memory_index[:] = load_index_from_file()
    else:
        print("Error: Index is empty after regeneration. Check data sources and processing logic.")

if __name__ == "__main__":
    # Check if the index file exists and is not empty
    if not os.path.exists(INDEX_FILE) or not load_index_from_file():
        print("Index file is missing or empty. Regenerating embeddings...")
        regenerate_index()
    else:
        # Always reload the index from file to ensure in-memory is up-to-date
        in_memory_index[:] = load_index_from_file()
        print("Index file found and loaded.")
        # Verify the loaded index contains both CSV and PDF data
        csv_items = sum(1 for item in in_memory_index if 'CSV' in item.get('source', ''))
        pdf_items = sum(1 for item in in_memory_index if 'PDF' in item.get('source', ''))
        print(f"Loaded index contains: {csv_items} CSV items, {pdf_items} PDF items")
        if pdf_items == 0:
            print("Warning: No PDF items found in index. Regenerating index...")
            regenerate_index()

    # Launch the UI
    print("Launching Gradio UI...")
    demo = create_ui()
    demo.launch()