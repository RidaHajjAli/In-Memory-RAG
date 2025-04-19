from .utils import get_embedding, calculate_similarity, get_generative_model  # Relative import
from typing import List, Dict, Any, Optional
import numpy as np
import config
import traceback  # Optional: for detailed error logging

def retrieve(query: str, index: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Retrieves the top_k most relevant items (text and source) from the index based on the query."""
    if not index:
        print("Error: Cannot retrieve from an empty index.")
        return []

    query_embedding = get_embedding(query)

    if query_embedding is None or all(e == 0.0 for e in query_embedding):
        print("Error: Could not generate embedding for the query.")
        return []

    context_embeddings = []
    valid_indices = []
    for i, item in enumerate(index):
        if (
            'embedding' in item and
            isinstance(item['embedding'], list) and
            len(item['embedding']) == config.EMBEDDING_DIMENSION and
            any(e != 0.0 for e in item['embedding'])
        ):
            context_embeddings.append(item['embedding'])
            valid_indices.append(i)

    if not context_embeddings:
        print("Error: No valid embeddings found in the index for retrieval.")
        return []

    context_embeddings_np = np.array(context_embeddings)
    if context_embeddings_np.ndim != 2 or context_embeddings_np.shape[1] != config.EMBEDDING_DIMENSION:
        print(f"Error: Context embeddings have unexpected shape: {context_embeddings_np.shape}. Expected (n, {config.EMBEDDING_DIMENSION}).")
        return []

    similarity_scores = calculate_similarity(query_embedding, context_embeddings_np)

    if similarity_scores.size == 0:
        print("Error: Similarity calculation failed or returned empty.")
        return []

    num_to_retrieve = min(top_k, len(similarity_scores))
    similarity_scores = np.nan_to_num(similarity_scores, nan=-2.0)
    top_k_indices_sorted = np.argsort(similarity_scores)[::-1][:num_to_retrieve]

    top_k_original_indices = [valid_indices[i] for i in top_k_indices_sorted if i < len(valid_indices)]
    retrieved_items = [index[i] for i in top_k_original_indices]

    print(f"Retrieved {len(retrieved_items)} documents.")
    return retrieved_items


def generate_answer(query: str, context_items: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Generates an answer using the LLM.
    If context_items is provided, it uses RAG prompt.
    If context_items is None or empty, it asks the LLM directly.
    """
    model = get_generative_model()

    if context_items:
        context_text = "\n\n".join([f"Source: {item['source']}\nContent: {item['text']}" for item in context_items])
        prompt = f"""
You are a helpful assistant answering questions about hospitals and the medical situation in Lebanon.
Answer the following question based *only* on the provided context below.
If the context doesn't contain the necessary information, clearly state that the provided context does not contain the answer. Do not use prior knowledge unless the context explicitly refers to it.

Context:
---
{context_text}
---

Question: {query}

Answer:
"""
        print("Generating answer using RAG context...")
    else:
        prompt = f"""
You are a helpful assistant with general knowledge. Please answer the following question about Lebanon to the best of your ability based on your internal knowledge.

Question: {query}

Answer:
"""
        print("Generating answer directly (without RAG context)...")

    try:
        response = model.generate_content(prompt)

        # Attempt to access finish reason safely
        actual_finish_reason = getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None
        reason_for_print = getattr(actual_finish_reason, 'name', str(actual_finish_reason)) if actual_finish_reason else "UNKNOWN"

        if response and hasattr(response, 'parts'):
            extracted_text = getattr(response, 'text', None)
            if not extracted_text:
                try:
                    extracted_text = response.candidates[0].content.parts[0].text
                except (AttributeError, IndexError):
                    pass

            if extracted_text:
                if actual_finish_reason and getattr(actual_finish_reason, 'name', '') != 'STOP':
                    return f"{extracted_text} (Note: Generation finished with reason: {reason_for_print})"
                return extracted_text

        if hasattr(response, 'prompt_feedback') and getattr(response.prompt_feedback, 'block_reason', None):
            block_reason = getattr(response.prompt_feedback.block_reason, 'name', str(response.prompt_feedback.block_reason))
            print(f"Warning: Response was blocked. Block Reason: {block_reason}. Feedback: {response.prompt_feedback}")
            return f"I apologize, the response was blocked (Reason: {block_reason}). Please try rephrasing your query."

        print(f"Warning: Received an unexpected or empty response structure. Finish Reason: {reason_for_print}. Response: {response}")
        return "I apologize, I received an unexpected response from the language model."

    except Exception as e:
        print(f"Error during generation: {e}")
        return "An error occurred while generating the answer."


def query_rag_pipeline(user_query: str, index: List[Dict[str, Any]], top_k: int) -> str:
    """Performs the RAG pipeline: retrieve context and generate answer."""
    print("--- Running RAG Pipeline ---")
    retrieved_items = retrieve(user_query, index, top_k)
    if not retrieved_items:
        return "Could not retrieve relevant context from the documents to answer the question."

    answer = generate_answer(user_query, context_items=retrieved_items)
    print("--- RAG Pipeline Complete ---")
    return answer
