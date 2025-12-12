import pandas as pd
import json
import re

# --- CONFIG ---
INPUT_CSV_PATH = "education_output.csv"
CLEAN_SUMMARY_CSV_PATH = "education_summary_clean.csv"

# --- HELPER FUNCTIONS ---

def clean_json_string(escaped_json: str) -> dict:
    """
    Attempts to unescape and parse the JSON stored in the CSV cells,
    handling common issues like double-quotes and potential model output corruption.
    """
    if escaped_json is None:
        return {}
    # treat empty / whitespace-only as empty
    if isinstance(escaped_json, float):  # catch NaN if any slipped through
        return {}
    if str(escaped_json).strip() == "":
        return {}

    try:
        # Replace common double-quote CSV escaping
        cleaned = str(escaped_json).replace('""', '"')

        # Fix common corruption patterns
        cleaned = re.sub(r'}\s*\{', '},{', cleaned)
        cleaned = re.sub(r'\[\s*{', '[{', cleaned)
        cleaned = re.sub(r'}\s*]', '}]', cleaned)
        cleaned = re.sub(r'""\s*""', '"", ""', cleaned)

        # Try to parse
        parsed_array = json.loads(cleaned)

        # Return the dictionary for the first item (if list) or the dict itself
        if isinstance(parsed_array, list) and parsed_array:
            return parsed_array[0]
        elif isinstance(parsed_array, dict):
            return parsed_array

    except Exception:
        # parsing failed -> return empty dict (caller will fallback)
        return {}

    return {}

# --- MAIN PROCESSING ---

def normalize_data(input_path: str, output_path: str):
    print(f"Loading data from {input_path}...")
    try:
        # Read as strings and avoid converting empty strings to NaN
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"FATAL ERROR: Could not read CSV file. Please check file path and separator. Error: {e}")
        return

    # Ensure expected columns exist so code won't crash
    expected = [
        'Name', 'Organization/Law Firm Name', 'Address Line 1', 'Address Line 2',
        'City', 'State', 'Country', 'Zipcode', 'Phone Number', 'Reg Code',
        'Agent/Attorney', 'input_index', 'input_name',
        'Education_JSON', 'Education_Summary', 'Education_Sources', 'Validation_Flags', 'Raw_Model_Output'
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = ""

    # 1. Filter out the necessary original and extracted columns
    original_cols = [
        'Name', 'Organization/Law Firm Name', 'Address Line 1', 'Address Line 2',
        'City', 'State', 'Country', 'Zipcode', 'Phone Number', 'Reg Code',
        'Agent/Attorney', 'input_index', 'input_name'
    ]
    extracted_cols = ['Education_JSON', 'Education_Summary', 'Education_Sources', 'Validation_Flags', 'Raw_Model_Output']

    df_clean = df[original_cols + extracted_cols].copy()

    # 2. Extract and clean data from the Raw_Model_Output column
    print("Extracting clean education details from JSON objects...")

    # Initialize clean columns from the existing columns (keeps original if parsing fails)
    df_clean['Clean_Education_List'] = df_clean['Education_JSON'].fillna("")
    df_clean['Clean_Sources'] = df_clean['Education_Sources'].fillna("")

    for index, row in df.iterrows():
        raw_val = row.get('Raw_Model_Output', "")
        if isinstance(raw_val, str) and raw_val.strip() != "":
            result_obj = clean_json_string(raw_val)
            edu_list = result_obj.get('education', [])
            sources = result_obj.get('sources', [])

            # Convert lists to JSON strings; if empty, keep existing value
            try:
                if edu_list is None or edu_list == []:
                    # keep existing df_clean value
                    pass
                else:
                    df_clean.loc[index, 'Clean_Education_List'] = json.dumps(edu_list, ensure_ascii=False)
            except Exception:
                # On any failure, leave as-is
                pass

            try:
                if sources is None or sources == []:
                    pass
                else:
                    df_clean.loc[index, 'Clean_Sources'] = json.dumps(sources, ensure_ascii=False)
            except Exception:
                pass

    # 3. Write the final, clean CSV file
    final_columns = original_cols + ['Education_Summary', 'Clean_Education_List', 'Clean_Sources', 'Validation_Flags']

    df_clean.to_csv(output_path, index=False, columns=final_columns, encoding='utf-8')
    print(f"\nSuccessfully created clean summary file at: {output_path}")


if __name__ == "__main__":
    normalize_data(INPUT_CSV_PATH, CLEAN_SUMMARY_CSV_PATH)
