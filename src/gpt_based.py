import os
import pandas as pd
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
RESULTS_DIR = f"results/gpt_only/{DATASET_NAME}"

# Ensure necessary directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load input texts and matching pool
debug_print("Loading input texts and matching pool...")
input_texts = pd.read_csv(INPUT_TEXTS_PATH)
matching_pool = pd.read_csv(MATCHING_POOL_PATH)

# Set up OpenAI API
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found! Set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=str(api_key))

# Initialize token usage tracking
total_input_tokens_used = 0
total_output_tokens_used = 0

# Function to find best match using GPT
def find_best_match_gpt(input_text, matching_pool):
    global total_input_tokens_used, total_output_tokens_used
    
    prompt = (
        f"Given the input '{input_text}', select the most appropriate match from the following options: {', '.join(matching_pool)}. "
        "Ensure that the selected match is not identical to the input. "
        "Provide only the best match and a short reasoning (8-10 words). "
        "Format the response as: {Best found match} - Reasoning."
    )
    
    response = client.chat.completions.create(
        model="o1-mini",
        messages=[{"role": "user", "content": prompt}],
        # max_tokens=20,
        # temperature=0.7
    )
    
    # Track token usage
    total_input_tokens_used += response.usage.prompt_tokens
    total_output_tokens_used += response.usage.completion_tokens
    debug_print(f"Cumulative Input Tokens: {total_input_tokens_used}, Cumulative Output Tokens: {total_output_tokens_used}")
    
    output = response.choices[0].message.content.strip()
    if "-" in output:
        best_match, reasoning = output.split("-", 1)
        return best_match.strip(), reasoning.strip()
    else:
        return output.strip(), "No reasoning provided."

# Process each input text
results = []
for _, row in input_texts.iterrows():
    best_match, reasoning = find_best_match_gpt(row["Input_Text"], matching_pool["Matched_Text"].tolist())
    results.append([row["Input_Text"], best_match, reasoning])

debug_print("Processed all entries.")

# Save results
output_file = os.path.join(RESULTS_DIR, "gpt_only_matching_results.csv")
results_df = pd.DataFrame(results, columns=["Input_Text", "Algorithm_Match", "Reasoning"])
results_df.to_csv(output_file, index=False)

debug_print(f"Results saved to {output_file}")
print("GPT-only similarity matching complete!")
