import time
import logging
import os
import pandas as pd
import tenacity
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- CONFIGURATION ---
INPUT_FILE = 'input_data.xlsx'
OUTPUT_FILE = 'education_output.csv'
# Let's try the Pro model, it is often more stable with Search tools than Flash
MODEL = 'gemini-2.5-flash' 
SLEEP_SECONDS = 4

# Setup Logging
logging.basicConfig(
    filename='education_extraction.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("CRITICAL ERROR: No GEMINI_API_KEY found in environment variables.")
    exit()

client = genai.Client(api_key=api_key)

# --- HELPER FUNCTIONS ---

def format_name_for_search(raw_name):
    if pd.isna(raw_name):
        return ""
    parts = str(raw_name).strip().split()
    if len(parts) >= 2:
        return f"{' '.join(parts[1:])} {parts[0]}"
    return raw_name

PROMPT_TEMPLATE = """
You are a precise data researcher.
Target: {name}
Organization: {org}
Location: {city}, {state}

1.  **SEARCH** for this person's biography (LinkedIn, Law Firm Bio, or Company Profile).
2.  **EXTRACT** their full education history.
3.  **VERIFY** the person matches the organization and location provided.

Output format (Strict):
[Degree] in [Major] - [Institution] ([Year]); [Degree] - [Institution] ([Year])

If no education data is found, output: "Data not available".
"""

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=30),
    retry=tenacity.retry_if_exception_type(Exception),
    reraise=True
)
def get_education(name, org, city, state):
    search_name = format_name_for_search(name)
    prompt = PROMPT_TEMPLATE.format(name=search_name, org=org, city=city, state=state)
    
    try:
        # Configuration for Google Search Tool
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_modalities=["TEXT"]
            )
        )
        
        # Check if we got a valid text response
        if response.text:
            return response.text.strip()
        else:
            # Sometimes response comes back but text is empty if blocked
            return "Error: Empty response from API"
            
    except Exception as e:
        # ERROR UNMASKING: This will print the exact reason to your console
        print(f"!!! API ERROR for {name}: {str(e)}")
        
        if "429" in str(e) or "Resource exhausted" in str(e):
            raise e 
        
        # Return the actual error string to the CSV so we can read it
        return f"Error: {str(e)}"

def main():
    print(f"Loading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return

    df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    
    # Simple check to handle existing file vs new file
    if os.path.exists(OUTPUT_FILE):
        df_done = pd.read_csv(OUTPUT_FILE)
        start_index = len(df_done)
        print(f"Resuming from row {start_index + 1}...")
    else:
        headers = list(df.columns) + ['Education_Details']
        pd.DataFrame(columns=headers).to_csv(OUTPUT_FILE, index=False)
        start_index = 0

    total_rows = len(df)
    print(f"Processing {total_rows} rows...")

    for i in range(start_index, total_rows):
        row = df.iloc[i]
        
        raw_name = row.get('Name')
        org = row.get('Organization/Law Firm Name')
        city = row.get('City', '')
        state = row.get('State', '')

        if pd.isna(raw_name):
            continue

        formatted_name = format_name_for_search(raw_name)
        print(f"[{i+1}/{total_rows}] Searching: {formatted_name}...")

        try:
            edu_data = get_education(raw_name, org, city, state)
            
            # Print result to console so you can see if it worked immediately
            print(f"   > Result: {edu_data[:100]}...") 

            result_row = row.copy()
            result_row['Education_Details'] = edu_data
            pd.DataFrame([result_row]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            if "429" in str(e) or "Resource exhausted" in str(e):
                print("DAILY LIMIT REACHED. Stopping.")
                break
            print(f"   > Critical Failure: {e}")

    print("Done.")

if __name__ == "__main__":
    main()