from .utils import get_embedding, calculate_similarity, get_generative_model
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import config
import re
from textblob import TextBlob

def preprocess_query(query: str) -> List[str]:
    """Enhanced query preprocessing with better spelling variation handling"""
    model = get_generative_model()
    
    # First, generate common spelling variations
    spelling_prompt = f"""Given this query: "{query}"
    Generate common spelling variations and mistakes that might occur when typing this query.
    Consider:
    1. Common typos (e.g., missing letters, swapped letters)
    2. Common misspellings
    3. Phonetic variations
    4. Common keyboard typos (adjacent keys)
    
    Return ONLY the variations as a comma-separated list, no explanations.
    Example format: variation1, variation2, variation3"""
    
    try:
        # Get spelling variations
        spelling_response = model.generate_content(spelling_prompt)
        spelling_variations = [v.strip() for v in spelling_response.text.split(',')]
        
        # Now get semantic variations
        semantic_prompt = f"""Given this query about healthcare in Lebanon: "{query}"
        Generate 5-7 variations of this query that capture different aspects and phrasings.
        Consider:
        1. Different phrasings and word orders
        2. Related medical terms and synonyms
        3. Question variations (what, how, where, when, why)
        4. Formal and informal language
        5. Common abbreviations and their full forms
        6. Different ways to express the same medical concept
        
        Return ONLY the variations as a comma-separated list, no explanations.
        Example format: variation1, variation2, variation3, variation4, variation5
        
        Important: Include variations that might appear in both medical documents and general healthcare information."""
        
        semantic_response = model.generate_content(semantic_prompt)
        semantic_variations = [v.strip() for v in semantic_response.text.split(',')]
        
        # Combine all variations
        all_variations = spelling_variations + semantic_variations
        
        # Always include the original query
        if query not in all_variations:
            all_variations.append(query)
            
        # Add basic variations if we have too few
        if len(all_variations) < 5:
            # Add question variations
            if not query.lower().startswith(('what', 'how', 'where', 'when', 'why')):
                for prefix in ['what', 'how', 'where', 'when', 'why']:
                    all_variations.append(f"{prefix} {query}")
            
            # Add common medical prefixes
            medical_prefixes = ['medical', 'healthcare', 'health', 'clinical']
            for prefix in medical_prefixes:
                all_variations.append(f"{prefix} {query}")
        
        # Remove duplicates while preserving order
        seen = set()
        variations = [x for x in all_variations if not (x in seen or seen.add(x))]
        
        print(f"Generated {len(variations)} query variations")
        return variations
    except Exception as e:
        print(f"Error in query preprocessing: {str(e)}")
        return [query]  # Fallback to original query

def retrieve(query: str, index: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Improved retrieval with comprehensive document coverage and spelling tolerance"""
    if not index:
        print("Error: Cannot retrieve from an empty index.")
        return []

    # Get query variations using Gemini
    query_variations = preprocess_query(query)
    print(f"Generated query variations: {query_variations}")
    
    # Get embeddings for all variations
    query_embeddings = []
    for variation in query_variations:
        embedding = get_embedding(variation)
        if embedding is not None and any(e != 0.0 for e in embedding):
            query_embeddings.append(embedding)
    
    if not query_embeddings:
        print("Error: Could not generate embeddings for the query.")
        return []

    # Calculate average embedding for more robust matching
    avg_query_embedding = np.mean(query_embeddings, axis=0)

    # Prepare context embeddings and track source types
    context_embeddings = []
    valid_indices = []
    keyword_matches = set()
    source_types = {}  # Track source types for debugging
    source_indices = {'CSV': [], 'PDF': [], 'Word': [], 'Excel': [], 'Other': []}  # Track indices by source type
    
    # Preprocess query for fuzzy matching
    query_words = set(query.lower().split())
    
    for i, item in enumerate(index):
        if 'embedding' in item and any(e != 0.0 for e in item['embedding']):
            context_embeddings.append(item['embedding'])
            valid_indices.append(i)
            source = item.get('source', 'Unknown')
            
            # Enhanced source type detection
            if source.startswith('CSV Row') or source.startswith('Uploaded CSV:'):
                source_type = 'CSV'
            elif source.startswith('PDF Document') or source.startswith('Uploaded PDF:'):
                source_type = 'PDF'
            elif source.startswith('Uploaded Word:'):
                source_type = 'Word'
            elif source.startswith('Uploaded Excel:'):
                source_type = 'Excel'
            else:
                source_type = 'Other'
            
            source_types[i] = source_type
            source_indices[source_type].append(i)
            
            # Enhanced keyword matching with fuzzy matching
            text = item['text'].lower()
            text_words = set(text.split())
            
            # Check for exact matches
            if any(variation.lower() in text for variation in query_variations):
                keyword_matches.add(i)
                print(f"Found exact keyword match in {source}")
            # Check for fuzzy matches (words that are close to query words)
            else:
                # Calculate word overlap
                word_overlap = len(query_words.intersection(text_words))
                if word_overlap >= len(query_words) * 0.7:  # 70% word overlap threshold
                    keyword_matches.add(i)
                    print(f"Found fuzzy keyword match in {source}")

    if not context_embeddings:
        print("Error: No valid embeddings found in the index for retrieval.")
        return []

    # Calculate similarities
    similarities = calculate_similarity(avg_query_embedding, np.array(context_embeddings))
    
    # Enhanced scoring system with better handling of spelling variations
    for idx in keyword_matches:
        if idx in valid_indices:
            pos = valid_indices.index(idx)
            # Boost exact matches more than fuzzy matches
            if any(variation.lower() in index[idx]['text'].lower() for variation in query_variations):
                similarities[pos] += 0.4  # Higher boost for exact matches
            else:
                similarities[pos] += 0.2  # Lower boost for fuzzy matches
            
            # Extra boost for uploaded files
            if source_types[idx] in ['PDF', 'Word', 'Excel']:
                similarities[pos] += 0.2

    # Ensure representation from each source type
    final_items = []
    remaining_slots = top_k
    min_items_per_source = max(1, top_k // len([s for s in source_indices if source_indices[s]]))
    
    # First pass: Get top items from each source type
    for source_type, indices in source_indices.items():
        if not indices:
            continue
            
        # Get similarities for this source type
        source_similarities = [similarities[valid_indices.index(i)] for i in indices]
        source_top_indices = np.argsort(source_similarities)[-min_items_per_source:][::-1]
        
        # Add top items from this source
        for idx in source_top_indices:
            if len(final_items) < top_k:
                original_idx = indices[idx]
                final_items.append(index[original_idx])
                remaining_slots -= 1

    # Second pass: Fill remaining slots with highest similarity items
    if remaining_slots > 0:
        remaining_indices = [i for i in range(len(similarities)) if index[valid_indices[i]] not in final_items]
        remaining_similarities = similarities[remaining_indices]
        remaining_top_indices = np.argsort(remaining_similarities)[-remaining_slots:][::-1]
        
        for idx in remaining_top_indices:
            original_idx = valid_indices[remaining_indices[idx]]
            final_items.append(index[original_idx])

    # Debugging information
    print(f"Retrieved {len(final_items)} items. Sources: {[item['source'] for item in final_items]}")
    
    # Print source type distribution
    source_distribution = {}
    for item in final_items:
        source = item['source']
        if 'CSV' in source:
            source_distribution['CSV'] = source_distribution.get('CSV', 0) + 1
        elif 'PDF' in source:
            source_distribution['PDF'] = source_distribution.get('PDF', 0) + 1
        elif 'Word' in source:
            source_distribution['Word'] = source_distribution.get('Word', 0) + 1
        elif 'Excel' in source:
            source_distribution['Excel'] = source_distribution.get('Excel', 0) + 1
        else:
            source_distribution['Other'] = source_distribution.get('Other', 0) + 1
    
    print(f"Source distribution in retrieved items: {source_distribution}")

    return final_items

def generate_answer(
    query: str, 
    context_items: Optional[List[Dict[str, Any]]] = None,
    conversation_history: Optional[str] = None,
    direct_llm: bool = False
) -> str:
    """Improved answer generation with better context handling"""
    model = get_generative_model()
    
    # Clean and normalize the query
    clean_query = query.strip()
    
    if direct_llm:
        # Direct LLM answer without context
        prompt = f"""
You are a helpful assistant with general knowledge. Please answer the following question to the best of your ability based on your internal knowledge.

Question: {clean_query}

Answer:
"""
        print("Generating direct LLM answer...")
    else:
        # Build context-aware prompt
        prompt_sections = [
            "You are a healthcare assistant for Lebanon. Answer precisely using the context:",
            "--- Conversation History ---",
            conversation_history or "No relevant history",
            "--- Context Documents ---"
        ]
        
        if context_items:
            for i, item in enumerate(context_items, 1):
                prompt_sections.append(f"[Document {i} from {item['source']}]\n{item['text']}")
        else:
            prompt_sections.append("No relevant documents found. Answer based on your internal knowledge.")
        
        prompt_sections += [
            "--- Question ---",
            clean_query,
            "##Instructions:",
            "1. If context documents are provided, answer using ONLY the documents.",
            "2. If no documents are provided, answer based on your internal knowledge.",
            "3. Be specific and precise in your answer.",
            "4. If the information is not available, say 'I couldn't find this information.'",
            "Answer:"
        ]
        
        prompt = "\n\n".join(prompt_sections)
        print("Generating answer with context...")

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def query_rag_pipeline(
    user_query: str,
    index: List[Dict[str, Any]],
    top_k: int,
    conversation_history: Optional[str] = None
) -> Tuple[str, List[str]]:
    """Enhanced RAG pipeline with better query handling"""
    # Process query and retrieve relevant items
    retrieved_items = retrieve(user_query, index, top_k)
    
    # Generate answer
    answer = generate_answer(
        query=user_query,
        context_items=retrieved_items,
        conversation_history=conversation_history
    )
    
    
    return answer, []