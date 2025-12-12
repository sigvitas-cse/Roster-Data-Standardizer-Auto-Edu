import time
import re
import logging
from google import genai
from dotenv import load_dotenv
import os
import pandas as pd
import tenacity

# Set up logging
logging.basicConfig(
    filename='processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()
client = genai.Client()

# Config
INPUT_FILE = 'input_data.xlsx'
OUTPUT_FILE = 'standardized_output.csv'
BATCH_SIZE = 1000
MODEL = 'gemini-2.5-flash'
DAILY_LIMIT = 250
SLEEP_SECONDS = 6

# Prompt template
PROMPT_TEMPLATE = """You are an expert in corporate and legal entity name standardization, with knowledge of official names from SEC filings, USPTO records, and company websites up to 2025. For each name below, output ONLY the official, standardized full name. Rules:
- Use proper capitalization and punctuation.
- Include standard suffixes like Inc., LLC, LLP if official (add LLP for law firms, LLC for companies if missing; keep AS for non-US firms).
- Remove extras like "Law Office of", "PC & Affiliates", or duplicates.
- Standardize spelling (e.g., "Aerovironment" → "AeroVironment").
- For solo practitioners (e.g., "John Doe Attorney"), use "John Doe" unless a firm name is clear.
- If ambiguous, choose the most common/official variant.
- Output format: Numbered list, e.g., "1. AeroVironment, Inc."

Examples:
Input: Google Inc.
Output: 1. Google LLC

Input: Patent Law Office of Joseph P. Abate
Output: 2. Joseph P. Abate

Input: Aerovironment, Inc.
Output: 3. AeroVironment, Inc.

Names to standardize:\n"""

@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    retry=tenacity.retry_if_exception_type(Exception)
)
def call_api(prompt):
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        return response.text
    except Exception as e:
        logging.error(f"API call failed: {e}")
        raise

# Load data from Excel
logging.info(f"Starting process for {INPUT_FILE}")
df = pd.read_excel(INPUT_FILE, engine='openpyxl')
total_rows = len(df)
logging.info(f"Loaded {total_rows} rows from {INPUT_FILE}")

# Process
processed = 0
daily_requests = 0
results = []

for i in range(0, total_rows, BATCH_SIZE):
    if daily_requests >= DAILY_LIMIT:
        logging.warning(f"Hit daily limit of {DAILY_LIMIT} requests. Process paused.")
        print("Hit daily limit. Resume tomorrow or upgrade tier.")
        break

    batch_df = df.iloc[i:i + BATCH_SIZE]
    org_names = batch_df['Organization/Law Firm Name'].tolist()  # Corrected column name
    numbered_batch = [f"{idx+1}. {name}" for idx, name in enumerate(org_names)]
    prompt = PROMPT_TEMPLATE + "\n".join(numbered_batch)

    logging.info(f"Processing batch {i//BATCH_SIZE + 1} (rows {i} to {i + len(batch_df) - 1})")
    print(f"Processing batch {i//BATCH_SIZE + 1} (rows {i} to {i + len(batch_df) - 1})")

    try:
        response_text = call_api(prompt)
        parsed = dict(re.findall(r'(\d+)\.\s*(.+)', response_text))
        for idx, row in batch_df.iterrows():
            orig_idx = idx - i + 1
            std_name = parsed.get(str(orig_idx), row['Organization/Law Firm Name'])  # Fallback to original
            new_row = row.copy()
            new_row['Organization/Law Firm Name(Output)'] = std_name.strip()  # Output column
            results.append(new_row)
        processed += len(batch_df)
        logging.info(f"Completed batch {i//BATCH_SIZE + 1}. Processed {processed}/{total_rows} rows.")
        print(f"Completed batch {i//BATCH_SIZE + 1}. Processed {processed}/{total_rows} rows.")
        daily_requests += 1
        time.sleep(SLEEP_SECONDS)
    except Exception as e:
        logging.error(f"Error in batch {i//BATCH_SIZE + 1}: {e}")
        print(f"Error in batch {i//BATCH_SIZE + 1}: {e}")
        results.extend(batch_df.to_dict('records'))

# Save with original and output columns
logging.info(f"Saving results to {OUTPUT_FILE}")
output_df = pd.DataFrame(results)
output_df.to_csv(OUTPUT_FILE, index=False)
logging.info(f"Process completed. Total rows processed: {processed}")
print(f"Done! Check {OUTPUT_FILE} and processing.log")