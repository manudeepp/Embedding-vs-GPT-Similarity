import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from openai import OpenAI

# Debug mode toggle
DEBUG_MODE = True
CONFIDENCE_THRESHOLD = 0.67  # Include matches within a set percentage of the highest similarity score

def debug_print(message):
    if DEBUG_MODE:
        print(message)

# Define dataset paths
DATASET_NAME = "wordsim_similarity_goldstandard"
INPUT_TEXTS_PATH = f"evaluation_data/{DATASET_NAME}/input_texts.csv"
MATCHING_POOL_PATH = f"evaluation_data/{DATASET_NAME}/matching_pool.csv"
EMBEDDING_DIR = f"embeddings/{DATASET_NAME}"
EMBEDDING_RESULTS_PATH = f"results/embedding_based/{DATASET_NAME}/embedding_top_matches.csv"
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

# Store top matches within confidence threshold in a dictionary
filtered_matches_dict = {}
for _, row in input_texts.iterrows():
    similarities = cosine_similarity([row["Embedding"]], all_matching_embeddings)[0] * 10
    sorted_indices = np.argsort(similarities)[::-1]
    highest_score = next((similarities[i] for i in sorted_indices if matching_pool.iloc[i]["Matched_Text"].lower() != row["Input_Text"].lower()), similarities[sorted_indices[0]])
    threshold = highest_score * CONFIDENCE_THRESHOLD
    
    filtered_matches_dict[row["Input_Text"]] = {
        "matches": [(matching_pool.iloc[i]["Matched_Text"], round(similarities[i], 2)) for i in sorted_indices if similarities[i] >= threshold]
    }

# Convert dictionary to DataFrame for saving
embedding_top_matches_df = pd.DataFrame([
    [input_text, json.dumps(matches["matches"])] for input_text, matches in filtered_matches_dict.items()
], columns=["Input_Text", "Embedding_Top_Matches"])
embedding_top_matches_df.to_csv(EMBEDDING_RESULTS_PATH, index=False)

debug_print(f"Top similarity matches saved to {EMBEDDING_RESULTS_PATH}")

# Load GPT-only results
gpt_only_results = pd.read_csv(GPT_ONLY_RESULTS_PATH)

# Compare if GPT match appears in threshold-filtered matches
comparison_df = gpt_only_results.merge(embedding_top_matches_df, on="Input_Text")
comparison_df.rename(columns={"Algorithm_Match": "Algorithm_Match_GPT"}, inplace=True)
comparison_df["GPT_In_Threshold_Embedding"] = comparison_df.apply(
    lambda row: row.get("Algorithm_Match_GPT", None) in [match for match, _ in json.loads(row["Embedding_Top_Matches"])], axis=1
)

# Compute percentage of matches where GPT's selected match is within the embedding threshold
match_percentage = (comparison_df["GPT_In_Threshold_Embedding"].sum() / len(comparison_df)) * 100

print("-" * 70)
print(f"GPT's match was found within embedding threshold {match_percentage:.2f}% of the time")
print("-" * 70)

# Save comparison results
output_file = os.path.join(COMPARISON_RESULTS_PATH, "embedding_threshold_vs_gpt_comparison.csv")
comparison_df.to_csv(output_file, index=False)

debug_print(f"Comparison results saved to {output_file}")
print("Embedding confidence threshold vs GPT-only comparison complete!")
