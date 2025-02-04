# **Embedding vs GPT: A Cost-Effective Approach to Semantic Similarity**

## **Overview**
This project explores whether embeddings can approximate the performance of GPT in semantic similarity tasks while being significantly more cost-effective. The study compares two approaches:
- **Embedding-Based Approach:** Uses OpenAI’s `text-embedding-3-small` model to generate vector representations and match words based on cosine similarity.
- **GPT-Only Approach:** Uses OpenAI’s `o1-mini` reasoning model to directly pick the best semantic match.

Through systematic evaluation using the **WordSim-353** dataset, this project assesses the accuracy, efficiency, and cost-effectiveness of embeddings in comparison to GPT.

## **Project Structure**
```
📂 embedding-vs-gpt-similarity
│── 📂 embeddings/wordsim_similarity_goldstandard
│   ├── input_texts_embeddings.csv
│   └── matching_pool_embeddings.csv
│
│── 📂 evaluation_data/wordsim_similarity_goldstandard
│   ├── evaluation_data.csv
│   ├── input_texts.csv
│   └── matching_pool.csv
│
│── 📂 input_data
│   ├── sts_benchmark_headlines.tsv
│   └── wordsim_similarity_goldstandard.txt
│
│── 📂 results
│   ├── 📂 comparisons/wordsim_similarity_goldstandard
│   │   └── adaptive_confidence_threshold.csv
│   │
│   ├── 📂 embedding_based/wordsim_similarity_goldstandard
│   │   ├── embedding_similarity_results.csv
│   │   └── embedding_top_matches.csv
│   │
│   ├── 📂 gpt_only/wordsim_similarity_goldstandard
│   │   └── gpt_only_matching_results.csv
│
│── 📂 src
│   ├── adaptive_confidence_threshold.py
│   ├── embedding_based.py
│   ├── generate_eval_data.py
│   └── gpt_based.py
│
│── .gitignore
│── LICENSE
│── README.md
│── requirements.txt
```

## **Setup Instructions**
### **Step 1: Install Dependencies**
Ensure you have Python installed. Install the required packages using:
```sh
pip install -r requirements.txt
```

### **Step 2: Set Up OpenAI API Key**
You need an OpenAI API key for this project. Set it up as an environment variable.

#### **On Windows (PowerShell)**
```sh
$env:OPENAI_API_KEY="your-api-key-here"
```
#### **On Mac/Linux (Terminal)**
```sh
export OPENAI_API_KEY="your-api-key-here"
```

Verify by running:
```sh
echo $OPENAI_API_KEY
```

## **Running the Scripts**
### **1. Preprocessing & Dataset Preparation**
To generate evaluation data:
```sh
python src/generate_eval_data.py
```

### **2. Running the Embedding-Based Approach**
Computes cosine similarity using precomputed embeddings:
```sh
python src/embedding_based.py
```
Outputs:
- `embedding_similarity_results.csv`
- `embedding_top_matches.csv`

### **3. Running the GPT-Only Approach**
Uses GPT to directly match input words:
```sh
python src/gpt_based.py
```
Outputs:
- `gpt_only_matching_results.csv`

### **4. Evaluating Adaptive Confidence Threshold**
Finds the **confidence threshold** at which embeddings best approximate GPT:
```sh
python src/adaptive_confidence_threshold.py
```
Outputs:
- `adaptive_confidence_threshold.csv`

## **Findings & Insights**
- **Embeddings achieved high accuracy at a fraction of GPT’s cost.**
- **Only the top 25% of embedding-based matches were needed** to approximate GPT results.
- **Cost Comparison (per 1M tokens, based on OpenAI pricing):**
  - **Embeddings (text-embedding-3-small):** `$0.02`
  - **GPT (o1-mini):** `$1.10` (input), `$4.40` (output)
- **Total Tokens Used in GPT Approach:**
  - **Input Tokens:** `56,084`
  - **Output Tokens:** `96,702`
  - **Total GPT Cost:** **$0.49**
  - **Embedding-Based Cost:** **$0.00112**

## **Key Takeaways**
- Embeddings are **25x cheaper** than GPT-only methods.
- Even when embeddings produced different matches than GPT, they were often **still reasonable**.
- **Example Cases:**
  - **Bread → GPT: Cash (slang), Embeddings: Food (logical synonym)**
  - **Marathon → GPT: Sprint (race type), Embeddings: Kilometer (distance metric)**
  - **King → GPT: Tiger (metaphor), Embeddings: Queen (direct synonym)**
  - **Cucumber → GPT: Fruit (botanical), Embeddings: Cabbage (same food category)**

These results suggest that **embedding-based methods are viable, interpretable, and highly cost-effective for many real-world applications**.

## **Future Work**
- Exploring hybrid approaches where **GPT refines embedding-based matches**.
- Testing the model on **sentence-level similarity (STS Benchmark)**.
- Optimizing embedding similarity matching with **adaptive techniques**.

## **License**
This project is open-source and free to use. Please refer to `LICENSE` for details.

## **Disclaimer**
This is an experimental study and results may vary based on dataset selection and OpenAI model updates.
