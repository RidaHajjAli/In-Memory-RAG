import gradio as gr
from src.rag import retrieve, generate_answer, query_rag_pipeline, preprocess_query
from src.utils import load_index_from_file, INDEX_FILE, save_index_to_file
from src.ingest import process_uploaded_file
import config
import time
import json
import os

in_memory_index = load_index_from_file()
if not in_memory_index:
    print(f"WARNING: Index loaded from {INDEX_FILE} is empty or file not found.")
else:
    print(f"Loaded index with {len(in_memory_index)} items from {INDEX_FILE}.")


def format_response(rag_answer, llm_answer, sources, show_llm):
    formatted = f"{rag_answer}\n\n"
    if show_llm:
        formatted += f"\n\n---\n**Direct LLM Answer:**\n{llm_answer}"
    return formatted

def handle_query(query, history, show_llm):
    global in_memory_index
    # Always reload the index from file before answering to ensure up-to-date
    in_memory_index[:] = load_index_from_file()
    # Retrieval is always performed on the full, up-to-date index (including uploaded files)
    start_time = time.time()
    conversation_history = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history])
    try:
        rag_answer, _ = query_rag_pipeline(
            user_query=query,
            index=in_memory_index,  # Always the full index
            top_k=config.TOP_K,
            conversation_history=conversation_history
        )
        llm_answer = ""
        if show_llm:
            llm_answer = generate_answer(
                query=query,
                direct_llm=True
            )
        response = rag_answer
        if show_llm:
            response += f"\n\n---\n**Direct LLM Answer:**\n{llm_answer}"
        process_time = time.time() - start_time
        return f"{response}\n\n_Processed in {process_time:.2f}s_"
    except Exception as e:
        return f"Error processing query: {str(e)}"

def handle_file_upload(file):
    global in_memory_index
    try:
        if file is None:
            return "No file was uploaded."
        if not hasattr(file, 'name'):
            return "Invalid file object received."
        print(f"Processing uploaded file: {file.name}")
        new_data = process_uploaded_file(file.name)
        if not new_data:
            return f"No valid data could be extracted from '{file.name}'"
        # Remove duplicates based on text+source
        existing_keys = set((item['text'], item['source']) for item in in_memory_index)
        for item in new_data:
            if (item['text'], item['source']) not in existing_keys:
                in_memory_index.append(item)
        save_index_to_file(in_memory_index)
        # After upload, reload index from file to ensure all future queries (even in new chat) use the latest index
        in_memory_index[:] = load_index_from_file()
        csv_items = sum(1 for item in in_memory_index if 'CSV' in item.get('source', ''))
        pdf_items = sum(1 for item in in_memory_index if 'PDF' in item.get('source', ''))
        other_items = len(in_memory_index) - csv_items - pdf_items
        return (
            f"File '{file.name}' processed successfully. {len(new_data)} items added to the index.<br>"
            f"Current index: {csv_items} CSV items, {pdf_items} PDF items, {other_items} other items.<br>"
            f"<span style='color:green'><b>Uploaded documents are now included in search results. Try asking about your uploaded file!</b></span>"
        )
    except Exception as e:
        print(f"Error in handle_file_upload: {str(e)}")
        return f"Error processing file: {str(e)}"

def create_ui():

    custom_css = """
    .gradio-container { padding-top: 1rem !important; }
    footer { display: none !important; }
    """

    with gr.Blocks(title="AI Chat Assistant", theme=gr.themes.Soft(), css=custom_css) as demo:
        chat_context_history = gr.State([])

        gr.Markdown("# AI Chat Assistant")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot_display = gr.Chatbot(
                    label="Conversation",
                    height=450,
                    bubble_full_width=False,
                )
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Chat with your data... ",
                    lines=2, # Initial height
                    scale=7 # Give textbox more width relative to buttons
                )

                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary", scale=1)
                    new_chat_btn = gr.Button("New Chat", scale=1)

            with gr.Column(scale=1): # Sidebar column
                toggle_llm = gr.Checkbox(
                    label="Show Direct LLM Answer",
                    value=False,
                    info="Compare RAG with raw LLM response"
                )
                file_upload = gr.File(
                    label="Upload File (Excel, Word, PDF)",
                    file_types=[".pdf", ".docx", ".xlsx", ".csv"],
                    type="filepath"
                )
                upload_status = gr.Markdown("")

                gr.Markdown("### Session Info")
                stats_display = gr.Markdown("**Session started.**\n0 messages")

        def start_new_chat():
            print("--- Starting New Chat ---")
            # Also reload the index to ensure new chat uses latest documents
            global in_memory_index
            in_memory_index[:] = load_index_from_file()
            return [], [], "**New chat started.**\n0 messages"

        def update_stats(context_history):
            num_messages = len(context_history)
            return f"**Turns in session:** {num_messages}"

        def user_message(query, display_history):
            print(f"User message added to display: '{query}'")
            return "", display_history + [[query, None]]

        def bot_response(display_history, show_llm, context_history):
            if not display_history or display_history[-1][0] is None:
                 print("Error: Bot response called without user query in history.")
                 return display_history, context_history, update_stats(context_history)

            query = display_history[-1][0]
            print(f"Bot processing query: '{query}'")
            response_text = handle_query(query, context_history, show_llm)
            display_history[-1][1] = response_text
            new_context_history = context_history + [[query, response_text]]
            print(f"Context history updated. Length: {len(new_context_history)}")
            return display_history, new_context_history, update_stats(new_context_history)

        # --- Event Listeners ---
        # Pressing Enter in the Textbox triggers this
        query_input.submit(
             user_message,
            [query_input, chatbot_display],
            [query_input, chatbot_display],
            queue=False
        ).then(
            bot_response,
            [chatbot_display, toggle_llm, chat_context_history],
            [chatbot_display, chat_context_history, stats_display]
        )

        # Clicking the Submit Button triggers this
        submit_btn.click(
            user_message,
            [query_input, chatbot_display],
            [query_input, chatbot_display],
            queue=False
        ).then(
            bot_response,
            [chatbot_display, toggle_llm, chat_context_history],
            [chatbot_display, chat_context_history, stats_display]
        )

        new_chat_btn.click(
            start_new_chat,
            None,
            [chatbot_display, chat_context_history, stats_display],
            queue=False
        )

        # File upload listener with better error handling
        file_upload.change(
            handle_file_upload,
            inputs=file_upload,
            outputs=upload_status,
            api_name="upload_file"
        )

    return demo


# --- if __name__ == "__main__": block remains the same ---
if __name__ == "__main__":
    config.configure_api()

    if not os.path.exists(INDEX_FILE):
         print(f"Index file '{INDEX_FILE}' not found.")
    else:
        if not in_memory_index:
             print(f"WARNING: Index file {INDEX_FILE} exists but loading failed or it's empty.")

    if not in_memory_index:
         print(f"ERROR: Index '{INDEX_FILE}' is empty or could not be loaded. Cannot start UI without data. Exiting.")
         exit()

    print("Launching Gradio UI...")
    demo = create_ui()
    demo.launch(server_port=7860, share=False)