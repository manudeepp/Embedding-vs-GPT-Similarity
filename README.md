# Carbon Emission Calculation using OpenAI API

## Overview

This project uses OpenAI’s embedding and chat capabilities to help find the closest match of construction materials from a dataset based on a given input and calculate the associated embodied carbon emissions. It works by finding the most similar entry from a database using embeddings and GPT.

## Setup Instructions

### Step 1: Install Required Packages

Install the required Python packages by running:

```sh
pip install -r requirements.txt
```

### Step 2: Setting Up the OpenAI API Key

You need an OpenAI API key to run this project. Follow these steps to set it up:

1. Create a Global Environment Variable for OpenAI API Key

   To make the OpenAI API key available globally across all PowerShell sessions (including Visual Studio’s terminal), you need to set it as a system environment variable.

   #### Via System Properties:

   1. Press `Win + X` and select **System**.
   2. Click on **Advanced system settings** on the left-hand side.
   3. In the **System Properties** window, click the **Environment Variables** button at the bottom.
   4. In the **Environment Variables** window, under **System variables**, click **New**.
   5. Set the **Variable name** to `OPENAI_API_KEY` and the **Variable value** to your OpenAI API key (e.g., `your-openai-api-key-here`).
   6. Click **OK** to save and apply the changes.
   7. Close all open windows and restart any applications (including Visual Studio) to pick up the new environment variable.

2. Verify the key by running:
   ```sh
   echo $env:OPENAI_API_KEY
   ```

### Step 3: Running the Script

After setting up the API key, run the Python script:

```sh
python carbonEmissionCalc.py
```

### File Descriptions

- **carbonEmissionCalc.py**: The main Python script that performs the embedding, material matching, and embodied carbon calculations.
- **ice-db/modified_ice_db.xlsx**: The database containing the materials and associated carbon data.
- **input_cwb_data/Mock CWPs.xlsx**: The input data that the script uses to find the matching material and calculate the embodied carbon.

### Usage

The script takes input data from the Mock CWPs file, finds the closest matching material from the ICE database using the OpenAI API, and calculates the total embodied carbon based on the material’s attributes and the provided quantity.

### Debugging

There is a `DEBUG_MODE` toggle at the start of the script that can be used to enable or disable debug print statements for easier troubleshooting.

- Set `DEBUG_MODE = True` for detailed output and to see the debug print statements during the run.

### Important Note

- Ensure your `.xlsx` files are placed in the correct directories (`ice-db/` and `input_cwb_data/`).
- The API key is a sensitive credential—make sure it’s kept secure.

### License

This project is for demonstration purposes only. Ensure to check your OpenAI API usage limits.

### Disclaimer

This is a Proof of Concept (POC) project, and the dataset used is a simplified free version from the ICE Database. It might not be fully comprehensive, and reliability may be limited in comparison to a licensed dataset.
