import os
import google.generativeai as genai
import numpy as np
from typing import List, Any
import config


# Global variable to store the embedding model instance
embedding_model = None
# Global variable to store the generative model instance (moved here for potential reuse)
generative_model = None

import json

INDEX_FILE = "index.json"  # File to save/load the index

def save_index_to_file(index, filepath=INDEX_FILE):
    """Saves the in-memory index to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=4)
    print(f"Index saved to {filepath}.")

def load_index_from_file(filepath=INDEX_FILE):
    """Loads the in-memory index from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            index = json.load(f)
        print(f"Index loaded from {filepath}.")
        return index
    except FileNotFoundError:
        print(f"No saved index found at {filepath}. Starting fresh.")
        return []
    except Exception as e:
        print(f"Error loading index from {filepath}: {e}")
        return []

def get_embedding_model():
    """Initializes and returns the embedding model instance using config."""
    global embedding_model
    if embedding_model is None:
        embedding_model = genai.GenerativeModel(config.EMBEDDING_MODEL_NAME)
        print(f"Embedding model loaded: {config.EMBEDDING_MODEL_NAME}")
    return embedding_model

def get_generative_model():
    """Initializes and returns the generative model instance using config."""
    global generative_model
    if generative_model is None:
        generative_model = genai.GenerativeModel(config.GENERATIVE_MODEL_NAME)
        print(f"Generative model loaded: {config.GENERATIVE_MODEL_NAME}")
    return generative_model

def get_embedding(text: str) -> List[float]:
    """Generates embeddings for the given text."""
    model = get_embedding_model()
    zero_vector = [0.0] * config.EMBEDDING_DIMENSION # Use dimension from config
    try:
        if not text or text.isspace():
            print(f"Warning: Attempted to embed empty text. Returning zero vector.")
            return zero_vector

        result = genai.embed_content(
            model=model.model_name,
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        # Add basic validation for expected structure
        if 'embedding' in result and isinstance(result['embedding'], list):
            return result['embedding']
        else:
            print(f"Warning: Unexpected embedding result structure for text: '{text[:50]}...'. Result: {result}. Returning zero vector.")
            return zero_vector

    except Exception as e:
        print(f"Error generating embedding for text: '{text[:50]}...' - {e}")
        return zero_vector


def calculate_similarity(query_embedding: List[float], context_embeddings: List[List[float]]) -> np.ndarray:
    """Calculates cosine similarity between a query embedding and a list of context embeddings."""
    query_emb = np.array(query_embedding).reshape(1, -1)
    context_embs = np.array(context_embeddings)

    if context_embs.size == 0 or context_embs.shape[1] != query_emb.shape[1]:
         print("Warning: Invalid or empty context embeddings provided for similarity calculation.")
         # Ensure dimension matches config even for empty array
         return np.array([]).reshape(0, config.EMBEDDING_DIMENSION)

    # Normalize embeddings
    query_norm = np.linalg.norm(query_emb)
    context_norms = np.linalg.norm(context_embs, axis=1)

    epsilon = 1e-10
    query_norm = max(query_norm, epsilon)
    context_norms[context_norms < epsilon] = epsilon

    similarity_scores = np.dot(context_embs, query_emb.T).flatten() / (context_norms * query_norm)
    similarity_scores = np.clip(similarity_scores, -1.0, 1.0)

    return similarity_scores