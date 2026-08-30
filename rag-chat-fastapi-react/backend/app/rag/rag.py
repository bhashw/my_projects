import os
import re
import numpy as np
import uuid
import torch
import json
import openai


def get_chunks(file_path, tokenizer, separator=" ", para_seperator="\n\n", chunk_size=100):
    doc_chunks = {}

    # Extract clean file name without path or extension
    base = os.path.basename(file_path)
    sku = os.path.splitext(base)[0]

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()

        # Split text into paragraphs
        paragraphs = re.split(para_seperator, text)

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            chunks_parag = []
            words = re.split(separator, paragraph.strip())
            sub_chunk = ""

            for word in words:
                # Test length if we add the new word
                test_chunk = f"{sub_chunk}{separator}{word}".strip() if sub_chunk else word

                if len(tokenizer.tokenize(test_chunk)) <= chunk_size:
                    sub_chunk = test_chunk
                else:
                    # Save current sub_chunk and start new one with current word
                    if sub_chunk:
                        chunks_parag.append(sub_chunk)
                    sub_chunk = word

            # Append trailing chunk after word loop finishes
            if sub_chunk:
                chunks_parag.append(sub_chunk)

            # Assign UUIDs and store in master dictionary
            for chunk_item in chunks_parag:
                if chunk_item.strip():
                    chunk_id = str(uuid.uuid4())
                    doc_chunks[chunk_id] = {"text": chunk_item, "metadata": {"file_name": sku}}

        return doc_chunks

    except Exception as e:
        print(f"Error getting chunks from file: {e}")
        return None


def document_map_embedding(doc_chunks, tokenizer, model):
    """
    This function maps doc_chunks which contains text chunks with their embeddings.
    """
    dict_map_embeddings = {}  # Initialize a dictionary to hold all chunk embeddings

    for chunk_id, chunk_content in doc_chunks.items():
        # Extract the text content of the chunk
        chunk_text = chunk_content.get("text")  # Use .get to avoid key errors

        # Tokenize the chunk text
        inputs = tokenizer(chunk_text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            # Get the embeddings for the chunk text
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

        # Map the chunk ID with its embeddings
        dict_map_embeddings[chunk_id] = embeddings

    return dict_map_embeddings


def save_data(path, data):
    """
    This function saves data on path
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Save data on {path}")


def load_data(path):
    """
    This function loads data from path
    """
    if not os.path.exists(path):
        print(f"File '{path}' does not exist.")
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"Load data from {path}")
    return data


def query_compute_embeddings(query, tokenizer, model):
    """
    This function compute the embeddings of the input query
    """
    # Transform the query into tokens
    query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        # Get the embeddings of the query from its tokens
        query_embeddings = model(**query_inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

    return query_embeddings


def compute_cosinus_similarity(query_embeddings, chunk_embeddings):
    """
    This function computes the cosinus similarity score between two vectors.
    It takes into parameters query and chunk embeddings's vectors
    """
    # Compute the dot product
    dot_product = np.dot(query_embeddings, chunk_embeddings)

    # Normalize both query embeddings and chunks vectors
    query_embeddings_norm = np.linalg.norm(query_embeddings)
    chunk_embeddings_norm = np.linalg.norm(chunk_embeddings)

    # If one of the norms is null return 0
    if query_embeddings_norm == 0 or chunk_embeddings_norm == 0:
        return 0

    else:
        return dot_product / (query_embeddings_norm * chunk_embeddings_norm)


def select_top_k_chunks(query_embeddings, dict_map_documents, top_k=3):
    """
    This function selects the top_k relevent chunks from the vector database based on the score similarity
    between each chunk's and query's embeddings.
    """
    # Dictionary to store similarity scores
    dict_similarity_score = {}
    for doc_id, doc_content in dict_map_documents.items():
        for chunk_id, chunk_embeddings in doc_content.items():
            # For each doc and chunk compute the similarity score
            dict_similarity_score[(doc_id, chunk_id)] = compute_cosinus_similarity(query_embeddings, chunk_embeddings)

    # Sort the dictionary and select the top_k chunks with the highest scores
    retreived_top_k = sorted(dict_similarity_score.items(), key=lambda item: item[1], reverse=True)[:top_k]

    # Get their correspondings texts
    top_k_result = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in retreived_top_k]

    return top_k_result


def retreive_chunks_content(top_k_result, dict_doc_store):
    """
    Retrieve ALL top-k chunks' text, not just the single best one.

    FIX: the previous version only read top_k_result[0], so despite
    computing top_k=3 matches upstream, only 1 chunk's text ever made
    it into the LLM prompt. This now joins every retrieved chunk's
    text together, so the model actually gets the full top-k context
    it was supposed to.
    """
    if not top_k_result:
        return None

    combined_text_parts = []
    for doc_id, chunk_id, score in top_k_result:
        chunk_entry = dict_doc_store.get(doc_id, {}).get(chunk_id)
        if chunk_entry:
            combined_text_parts.append(chunk_entry["text"])

    if not combined_text_parts:
        return None

    # Return the same {"text": ...} shape generate_llm_response expects,
    # but now text is every retrieved chunk joined together.
    return {"text": "\n\n".join(combined_text_parts)}



client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_llm_response(query, relevent_chunks):
    """
    This function generates responses based on the query and retrieved chunks using OpenAI's API.
    """
    if relevent_chunks:
        # Create the prompt using the provided query and relevant chunks
        template = f"""
        You are an intelligent search engine. You will be provided with some retrieved context, as well as the user's query.

        Your job is to understand the request and answer based on the retrieved context. If you can't find the answer, you can say "I don't know".

        Here is the retrieved context:
        {relevent_chunks['text']}

        Here is the user's query:
        {query}
        """
    else:
        # Create the prompt using the provided query and relevant chunks
        template = f"""
        You are an intelligent search engine. Your job is to understand the request and answer based on your knowledge If you can't find the answer, you can say "I don't know".
        Here is the user's query:
        {query}
        """

    # Call OpenAI's API to generate a response for the given message
    # (new-style call: client.chat.completions.create, not openai.ChatCompletion.create)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": template}],
        max_tokens=200
    )

    
    bot_message = response.choices[0].message.content

    print("LLM Response:", bot_message)

    if not bot_message:
        print("No response generated by LLM.")

    return bot_message