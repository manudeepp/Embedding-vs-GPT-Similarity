import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
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
GPT_ONLY_RESULTS_PATH = f"results/gpt_only/{DATASET_NAME}/gpt_only_matching_results.csv"
COMPARISON_RESULTS_PATH = f"results/comparisons/{DATASET_NAME}"

# Ensure necessary directories exist
os.makedirs(COMPARISON_RESULTS_PATH, exist_ok=True)

# Load input texts and matching pool
debug_print("Loading input texts and matching pool...")
input_texts = pd.read_csv(INPUT_TEXTS_PATH)
matching_pool = pd.read_csv(MATCHING_POOL_PATH)

# Set up OpenAI API
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found! Set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=str(api_key))

# Function to load or compute embeddings
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

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

# Load or compute embeddings for input texts and matching pool
input_texts = generate_embeddings(input_texts, "Input_Text", "input_texts_embeddings.csv")
matching_pool = generate_embeddings(matching_pool, "Matched_Text", "matching_pool_embeddings.csv")

# Convert embeddings to numpy arrays
input_texts["Embedding"] = input_texts["Embedding"].apply(np.array)
matching_pool["Embedding"] = matching_pool["Embedding"].apply(np.array)
all_matching_embeddings = np.vstack(matching_pool["Embedding"].tolist())

# Store confidence threshold results
confidence_data = []

# Load GPT-only results
gpt_only_results = pd.read_csv(GPT_ONLY_RESULTS_PATH)

# Process each input text from GPT-only results
for _, row in gpt_only_results.iterrows():
    input_text = row["Input_Text"]
    gpt_match = row["Algorithm_Match"]
    input_embedding = input_texts[input_texts["Input_Text"] == input_text]["Embedding"].values
    
    if len(input_embedding) == 0:
        continue

    input_embedding = np.array(input_embedding[0]).reshape(1, -1)  # Ensure correct shape
    
    similarities = cosine_similarity(input_embedding, all_matching_embeddings)[0] * 10
    sorted_indices = np.argsort(similarities)[::-1]
    
    top_match_similarity = similarities[sorted_indices[0]]
    match_threshold = 1.0  # Start with full confidence
    
    for i in sorted_indices:
        if matching_pool.iloc[i]["Matched_Text"] == gpt_match:
            match_similarity = similarities[i]
            match_threshold = round(match_similarity / top_match_similarity, 2)
            break
    
    matched_list = [(matching_pool.iloc[i]["Matched_Text"], round(similarities[i], 2)) for i in sorted_indices if similarities[i] >= match_similarity]
    confidence_data.append([input_text, gpt_match, match_threshold, json.dumps(matched_list)])

# Compute final confidence metrics
confidence_df = pd.DataFrame(confidence_data, columns=["Input_Text", "GPT_Match", "Required_Confidence_Threshold", "Embedding_Matches"])
avg_confidence_threshold = confidence_df["Required_Confidence_Threshold"].mean()

# Save confidence threshold results
output_file = os.path.join(COMPARISON_RESULTS_PATH, "adaptive_confidence_threshold.csv")
confidence_df.to_csv(output_file, index=False)

debug_print(f"Comparison results saved to {output_file}")
print(f"Average Required Confidence Threshold: {avg_confidence_threshold:.4f}")
print("Adaptive Confidence Threshold evaluation complete!")
