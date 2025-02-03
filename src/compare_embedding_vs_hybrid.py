import os
import pandas as pd
import numpy as np

# Debug mode toggle
DEBUG_MODE = True

def debug_print(message):
    if DEBUG_MODE:
        print(message)

# Define dataset paths
DATASET_NAME = "wordsim_similarity_goldstandard"
EMBEDDING_RESULTS_PATH = f"results/embedding_based/{DATASET_NAME}/embedding_similarity_results.csv"
HYBRID_RESULTS_PATH = f"results/hybrid_embedding_gpt/{DATASET_NAME}/hybrid_embedding_gpt_results.csv"
GPT_ONLY_RESULTS_PATH = f"results/gpt_only/{DATASET_NAME}/gpt_only_matching_results.csv"
COMPARISON_RESULTS_PATH = f"results/comparisons/{DATASET_NAME}"

# Ensure necessary directories exist
os.makedirs(COMPARISON_RESULTS_PATH, exist_ok=True)

# Load results from all approaches
debug_print("Loading results from embedding, hybrid, and GPT-only approaches...")
embedding_results = pd.read_csv(EMBEDDING_RESULTS_PATH)
hybrid_results = pd.read_csv(HYBRID_RESULTS_PATH)
gpt_only_results = pd.read_csv(GPT_ONLY_RESULTS_PATH)

# Merge results on Input_Text
comparison_df = embedding_results.merge(hybrid_results, on="Input_Text", suffixes=("_Embedding", "_Hybrid"))
comparison_df = comparison_df.merge(gpt_only_results, on="Input_Text", suffixes=("", "_GPT"))
comparison_df.rename(columns={"Algorithm_Match": "Algorithm_Match_GPT"}, inplace=True)

# Match Agreement with GPT
comparison_df["Match_Agreement_Hybrid_vs_GPT"] = comparison_df["Algorithm_Match_Hybrid"] == comparison_df["Algorithm_Match_GPT"]
comparison_df["Match_Agreement_Embedding_vs_GPT"] = comparison_df["Algorithm_Match_Embedding"] == comparison_df["Algorithm_Match_GPT"]
comparison_df["Match_Agreement_Embedding_vs_Hybrid"] = comparison_df["Algorithm_Match_Embedding"] == comparison_df["Algorithm_Match_Hybrid"]

# Calculate match agreement percentages
total_comparisons = len(comparison_df)
hybrid_vs_gpt_match_percentage = (comparison_df["Match_Agreement_Hybrid_vs_GPT"].sum() / total_comparisons) * 100
embedding_vs_gpt_match_percentage = (comparison_df["Match_Agreement_Embedding_vs_GPT"].sum() / total_comparisons) * 100
embedding_vs_hybrid_match_percentage = (comparison_df["Match_Agreement_Embedding_vs_Hybrid"].sum() / total_comparisons) * 100

# Similarity Score Deviation for Hybrid vs Embedding
comparison_df["Similarity_Score_Deviation_Hybrid_vs_Embedding"] = abs(comparison_df["Similarity_Score_Embedding"] - comparison_df["Similarity_Score_Hybrid"])
avg_similarity_deviation = comparison_df["Similarity_Score_Deviation_Hybrid_vs_Embedding"].mean()

# Save comparison results
output_file = os.path.join(COMPARISON_RESULTS_PATH, "full_model_comparison.csv")
comparison_df.to_csv(output_file, index=False)

# Print results
debug_print(f"Comparison results saved to {output_file}")
debug_print(f"Hybrid vs GPT Match Agreement: {hybrid_vs_gpt_match_percentage:.2f}%")
debug_print(f"Embedding vs GPT Match Agreement: {embedding_vs_gpt_match_percentage:.2f}%")
debug_print(f"Embedding vs Hybrid Match Agreement: {embedding_vs_hybrid_match_percentage:.2f}%")
debug_print(f"Average Similarity Score Deviation (Hybrid vs Embedding): {avg_similarity_deviation:.4f}")

print("Comparison between embedding, hybrid, and GPT-only approaches complete!")
