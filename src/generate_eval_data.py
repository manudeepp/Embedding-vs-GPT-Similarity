import os
import pandas as pd

# Load dataset (WordSim-353 or STS Benchmark)
DATA_PATH = "input_data/wordsim_similarity_goldstandard.txt"  # Change as needed
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

# Extract dataset name
dataset_name = os.path.splitext(os.path.basename(DATA_PATH))[0]
EVAL_DIR = os.path.join("evaluation_data", dataset_name)
os.makedirs(EVAL_DIR, exist_ok=True)

print("Loading dataset...")
input_data = pd.read_csv(DATA_PATH, sep='\t', header=None)

# Detect dataset type and assign column names dynamically
if input_data.shape[1] == 3:
    input_data.columns = ["Input_Text", "Matched_Text", "Human_Score"]
else:
    raise ValueError("Unexpected file format. Expected three columns (Input_Text, Matched_Text, Human_Score).")

print("Filtering highest human score matches...")
# Select the highest-scoring match for each unique Input_Text
evaluation_data = input_data.loc[input_data.groupby("Input_Text")["Human_Score"].idxmax()]

# Extract all unique matched words as the "matching pool"
matching_pool = input_data["Matched_Text"].unique()
matching_pool_df = pd.DataFrame(matching_pool, columns=["Matched_Text"])

# Extract and save unique input words
unique_input_words = evaluation_data[["Input_Text"]].drop_duplicates()
input_words_file = os.path.join(EVAL_DIR, "input_texts.csv")
unique_input_words.to_csv(input_words_file, index=False)
print(f"Unique input words saved to {input_words_file}")

# Save evaluation input dataset
evaluation_file = os.path.join(EVAL_DIR, "evaluation_data.csv")
matching_pool_file = os.path.join(EVAL_DIR, "matching_pool.csv")

evaluation_data.to_csv(evaluation_file, index=False)
matching_pool_df.to_csv(matching_pool_file, index=False)

print(f"Evaluation dataset saved to {evaluation_file}")
print(f"Matching pool saved to {matching_pool_file}")
