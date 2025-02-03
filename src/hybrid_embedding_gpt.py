import os
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI

# Debug mode toggle
DEBUG_MODE = True

def debug_print(message):
    if DEBUG_MODE:
        print(message)

# Define dataset paths
DATASET_NAME = "wordsim_similarity_goldstandard"
INPUT_TEXTS_PATH = f"evaluation_data/{DATASET_NAME}/input_texts.csv"
MATCHING_POOL_PATH = f"evaluation_data/{DATASET_NAME}/matching_pool.csv"
EMBEDDING_DIR = f"embeddings/{DATASET_NAME}"
RESULTS_DIR = f"results/hybrid_embedding_gpt/{DATASET_NAME}"

# Ensure necessary directories exist
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load input words and matching pool
debug_print("Loading input words and matching pool...")
input_texts = pd.read_csv(INPUT_TEXTS_PATH)
matching_pool = pd.read_csv(MATCHING_POOL_PATH)

# Set up OpenAI API
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found! Set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=str(api_key))

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

# Function to load or compute embeddings
def generate_embeddings(data, column_name, embedding_file):
    embedding_path = os.path.join(EMBEDDING_DIR, embedding_file)
    if os.path.exists(embedding_path):
        embeddings_df = pd.read_csv(embedding_path)
        embeddings_df["Embedding"] = embeddings_df["Embedding"].apply(json.loads)
        debug_print(f"Loaded existing embeddings from {embedding_path}")
    else:
        debug_print(f"Generating embeddings for {column_name}...")
        embeddings_df = data.copy()
        embeddings_df["Embedding"] = embeddings_df[column_name].apply(get_embedding)
        embeddings_df.to_csv(embedding_path, index=False)
        debug_print(f"Embeddings saved to {embedding_path}")
    return embeddings_df

# Generate or load embeddings
input_embeddings = generate_embeddings(input_texts, "Input_Text", "input_texts_embeddings.csv")
matching_embeddings = generate_embeddings(matching_pool, "Matched_Text", "matching_pool_embeddings.csv")

# Convert embeddings to numpy arrays
input_embeddings["Embedding"] = input_embeddings["Embedding"].apply(np.array)
matching_embeddings["Embedding"] = matching_embeddings["Embedding"].apply(np.array)
all_matching_embeddings = np.vstack(matching_embeddings["Embedding"].tolist())

# Compute cosine similarity and find top 5 matches
def find_top_matches(input_embedding, all_embeddings, matched_texts, input_text, top_n=5):
    similarities = cosine_similarity([input_embedding], all_embeddings)[0] * 10
    sorted_indices = np.argsort(similarities)[::-1]
    top_matches = [(matched_texts[idx], similarities[idx]) for idx in sorted_indices if matched_texts[idx] != input_text][:top_n]
    return top_matches

# Use GPT to select the best match with reasoning
def refine_match_with_gpt(input_text, top_matches):
    candidates = "\n".join([f"{i+1}. {match[0]} (Score: {match[1]:.2f})" for i, match in enumerate(top_matches)])
    prompt = f"""
    Given the input text: "{input_text}", select the most appropriate match from the following options:
    {candidates}
    
    Respond with the best match number and a brief reasoning (8-10 words).
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    output = response.choices[0].message.content.strip()
    best_match_idx = output.split(".")[0].strip()
    reasoning = output[len(best_match_idx) + 2:].strip()
    best_match_idx = int(best_match_idx) - 1
    return top_matches[best_match_idx][0], top_matches[best_match_idx][1], reasoning

results = []
for _, row in input_embeddings.iterrows():
    top_matches = find_top_matches(row["Embedding"], all_matching_embeddings, matching_embeddings["Matched_Text"].tolist(), row["Input_Text"])
    best_match_text, best_match_score, reasoning = refine_match_with_gpt(row["Input_Text"], top_matches)
    results.append([row["Input_Text"], best_match_text, best_match_score, reasoning])

debug_print("Processed all entries.")

# Save results
output_file = os.path.join(RESULTS_DIR, "hybrid_embedding_gpt_results.csv")
results_df = pd.DataFrame(results, columns=["Input_Text", "Algorithm_Match", "Similarity_Score", "Reasoning"])
results_df.to_csv(output_file, index=False)
debug_print(f"Results saved to {output_file}")

print("Hybrid embedding + GPT similarity computation complete!")
