#!/usr/bin/env python3
"""
education_extraction_batch_final.py

Robust structured education extraction for people in input_data.xlsx using google.genai
with GoogleSearch tool, now using EFFICIENT BATCH PROCESSING (100 rows per request).

- Implements Variable Batches / Maximized Size strategy.
- Fixes NameError and implements buffered CSV writing for reliability.
- Requires: google-genai, requests, pandas, tenacity, python-dotenv
- Set GEMINI_API_KEY in environment (or .env).
"""

import os
import re
import time
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import tenacity
import requests
from google import genai
from google.genai import types

# ---------------- CONFIG ----------------
INPUT_FILE = "input_data.xlsx"
OUTPUT_FILE = "education_output.csv"
MANUAL_REVIEW_FILE = "manual_review.csv"
RESOLVED_URLS_FILE = "resolved_urls.csv"
LOG_FILE = "education_extraction_batch.log"

MODEL = "gemini-2.5-flash"
SLEEP_SECONDS = 15      # Time to wait BETWEEN BATCHES (to respect RPM limit)
RETRY_ATTEMPTS = 3
DAILY_LIMIT = 20        # Max Requests Per Day (RPD)
BATCH_SIZE = 100        # Max rows per request (aiming for 20 RPD * 100 rows/batch = 2000 daily limit)
TEST_COUNT = None       # change to None to run all rows

# ----------------------------------------

# Logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("Missing GEMINI_API_KEY in environment")

client = genai.Client(api_key=api_key)
current_year = datetime.now().year

# Strict prompt (MODIFIED FOR BATCH INPUT)
PROMPT_TEMPLATE = """
You are a careful, literal data researcher. For the list of people below, find authoritative public sources
(law firm bio, company bio, university profile, LinkedIn profile, or official directories) and extract
a complete education history for **each person**.

**IMPORTANT RULES (read and obey):**
1. Use EXACT input_index provided for each result. **Do not renumber** results.
2. Return STRICT JSON ONLY (no commentary) — a JSON array of objects (one object per input row).
3. The JSON array must contain exactly one object for every input item provided in the INPUT_DATA section.
4. Each object must be:
{
  "input_index": <integer, must match input>,
  "input_name": "<original name from input>",
  "education": [
    {
      "degree": "<e.g., J.D., LL.M., B.S., M.S., Ph.D.>",
      "field": "<field/major or empty>",
      "institution": "<full institution name>",
      "graduation_year": "<YYYY or empty>",
      "notes": "<honors/in progress/other or empty>",
      "source": "<canonical URL to the authoritative source or empty>"
    },
    ...
  ],
  "best_summary": "<one-sentence summary or 'No verified education info found'>",
  "sources": ["<canonical URLs used>"]
}

5. **Do NOT invent degrees or years.** If a degree is in progress, set graduation_year to "" and notes to "in progress/expected - cite source".
6. Return canonical page URLs (the real page URL).
7. If you cannot find authoritative sources for a person, set their "education": [], "sources": [], and "best_summary": "No verified education info found".

INPUT_DATA (List of people to process):
"""

@tenacity.retry(
    stop=tenacity.stop_after_attempt(RETRY_ATTEMPTS),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
    retry=tenacity.retry_if_exception_type(Exception),
    reraise=True
)
def call_model(prompt_text: str):
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt_text,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            response_modalities=["TEXT"]
        )
    )
    text = getattr(resp, "text", None)
    if text is None:
        text = str(resp)
    return text

def sanitize_model_text(text: str) -> str:
    """Remove code fences and unescape common escapes so JSON can be found."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("\n")
        if len(parts) >= 2:
            parts = parts[1:]
            if parts and parts[-1].strip().startswith("```"):
                parts = parts[:-1]
            t = "\n".join(parts).strip()
    t = t.replace('\\"', '"')
    t = t.replace('“', '"').replace('”', '"').replace("’", "'")
    return t

def extract_json_array(text: str):
    """Extract first JSON array substring and parse it. Return Python object or None."""
    t = sanitize_model_text(text)
    start = t.find('[')
    end = t.rfind(']')
    if start == -1 or end == -1 or end <= start:
        return None
    json_blob = t[start:end+1]
    json_blob = re.sub(r',\s*([\]\}])', r'\1', json_blob)
    try:
        parsed = json.loads(json_blob)
        return parsed
    except Exception as e:
        logging.exception("JSON parsing failed: %s", e)
        return None

def resolve_url(url: str, timeout=10):
    """Follow redirects and return (resolved_url, status_code). If error, return (original, None)."""
    try:
        cleaned = str(url).strip()
        tokens = cleaned.split()
        candidate = None
        for tok in tokens[::-1]:
            if tok.lower().startswith("http"):
                candidate = tok
                break
        if candidate is None:
            candidate = tokens[0]
        r = requests.get(candidate, timeout=timeout, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        return r.url, r.status_code
    except Exception as e:
        logging.warning("resolve_url error for %s: %s", url, str(e))
        return url, None

def validate_education_items(items):
    """
    items: list of dicts from model 'education' field.
    Returns (cleaned_items, flags) where cleaned_items are normalized and flags list issues.
    """
    cleaned = []
    flags = []
    for ent in (items or []):
        if not isinstance(ent, dict):
            flags.append("malformed_education_entry")
            continue
        degree = ent.get("degree", "") or ""
        field = ent.get("field", "") or ""
        institution = ent.get("institution", "") or ""
        grad_raw = ent.get("graduation_year", "") or ""
        notes = ent.get("notes", "") or ""
        source = ent.get("source", "") or ""

        year = ""
        if isinstance(grad_raw, int):
            year = str(grad_raw)
        elif isinstance(grad_raw, str):
            m = re.search(r'(19|20)\d{2}', grad_raw)
            if m:
                year = m.group(0)

        if year:
            try:
                y = int(year)
                if y > current_year:
                    flags.append(f"graduation_year_in_future:{year} for {institution}")
            except:
                pass

        if degree and not source:
            flags.append("degree_without_source")

        cleaned.append({
            "degree": degree.strip(),
            "field": field.strip(),
            "institution": institution.strip(),
            "graduation_year": year,
            "notes": notes.strip(),
            "source": source.strip()
        })
    return cleaned, flags

def find_name_column(df):
    if "Name" in df.columns:
        return "Name"
    for c in df.columns:
        if "name" in c.lower():
            return c
    return df.columns[0]

def append_row_to_csv(path, row_dict, columns):
    """Writes a single row (dict) to CSV, appending without header."""
    df = pd.DataFrame([row_dict])
    df.to_csv(path, mode="a", header=False, index=False, columns=columns, encoding="utf-8")

def initialize_csv_if_needed(path, columns):
    """Creates the CSV file with headers if it doesn't exist."""
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8")

# ---------------- CORE BATCHING LOGIC ----------------

def main():
    global DAILY_LIMIT, BATCH_SIZE

    logging.info("Starting batch extraction run")
    if not os.path.exists(INPUT_FILE):
        logging.error("Input file missing: %s", INPUT_FILE)
        raise SystemExit("Input file not found")

    df = pd.read_excel(INPUT_FILE, engine="openpyxl")
    name_col = find_name_column(df)
    total_rows = len(df)
    logging.info("Loaded %d rows; using name column: %s", total_rows, name_col)

    csv_columns = list(df.columns) + ["input_index", "input_name", "Education_JSON",
                                     "Education_Summary", "Education_Sources",
                                     "Validation_Flags", "Raw_Model_Output"]
    initialize_csv_if_needed(OUTPUT_FILE, csv_columns)

    manual_columns = csv_columns + ["Reason"]
    initialize_csv_if_needed(MANUAL_REVIEW_FILE, manual_columns)
    initialize_csv_if_needed(RESOLVED_URLS_FILE, ["original_url", "resolved_url", "status_code"])

    try:
        processed_df = pd.read_csv(OUTPUT_FILE, encoding="utf-8")
        processed_indices = set(processed_df.get("input_index", []).astype(str).tolist())
        logging.info("Resuming; found %d processed indices", len(processed_indices))
    except:
        processed_indices = set()

    # --- RESTORED LOGIC TO DEFINE total_to_process ---
    df_to_process = df[~df.index.to_series().apply(lambda x: str(x + 1) in processed_indices)].copy()
    total_to_process = len(df_to_process)
    logging.info("Rows remaining to process: %d", total_to_process)

    if TEST_COUNT and isinstance(TEST_COUNT, int):
        df_to_process = df_to_process.head(TEST_COUNT)
        total_to_process = len(df_to_process)
        logging.info("TEST_COUNT set: running first %d rows of remaining data", total_to_process)
    # --- END RESTORED LOGIC ---

    if total_to_process <= (DAILY_LIMIT * BATCH_SIZE):
        R_Ideal = (total_to_process + BATCH_SIZE - 1) // BATCH_SIZE
        R_Used = min(R_Ideal, DAILY_LIMIT)
        BATCH_SIZE = (total_to_process + R_Used - 1) // R_Used
        DAILY_LIMIT = R_Used

    logging.info("Batching Strategy: Using BATCH_SIZE=%d and DAILY_LIMIT=%d", BATCH_SIZE, DAILY_LIMIT)

    daily_calls = 0

    # --- Batch Buffers ---
    output_buffer = []
    manual_buffer = []
    
    # The for loop now runs without NameError
    for i in range(0, total_to_process, BATCH_SIZE):
        
        if daily_calls >= DAILY_LIMIT:
            logging.warning("Reached daily limit; stopping")
            print(f"\nReached daily limit ({DAILY_LIMIT} requests). Stop and resume later.")
            break
            
        batch_df = df_to_process.iloc[i:i + BATCH_SIZE]
        batch_start_index = batch_df.index[0] + 1
        batch_end_index = batch_df.index[-1] + 1

        batch_input_data = []
        for original_idx, row in batch_df.iterrows():
            input_index = original_idx + 1
            name = str(row.get(name_col, "")).strip()
            org = str(row.get("Organization/Law Firm Name", "") or "").strip()
            city = str(row.get("City", "") or "").strip()
            state = str(row.get("State", "") or "").strip()

            batch_input_data.append(
                f"[{input_index}] Name: {name} | Organization: {org} | Location: {city}, {state}"
            )

        prompt = PROMPT_TEMPLATE + "\n" + "\n".join(batch_input_data) + "\n\nReturn JSON now."

        print(f"\n[Request {daily_calls + 1}/{DAILY_LIMIT}] Processing batch {batch_start_index}-{batch_end_index} ({len(batch_df)} rows)...")

        
        batch_success_count = 0
        batch_manual_count = 0
        
        # Clear buffers for the new batch
        output_buffer.clear()
        manual_buffer.clear() 

        try:
            logging.info("Calling model for batch: %d to %d", batch_start_index, batch_end_index)
            model_text = call_model(prompt)
            model_text_s = sanitize_model_text(model_text)
            parsed_results = extract_json_array(model_text_s)
            
            if parsed_results is None:
                # Catastrophic JSON failure logic: write rows to manual review buffer
                logging.error("CRITICAL: Failed to parse JSON array for batch %d-%d. Output sent to manual review buffer.", batch_start_index, batch_end_index)
                for original_idx, row in batch_df.iterrows():
                    input_index = original_idx + 1
                    out_row = row.copy()
                    out_row["input_index"] = input_index
                    out_row["input_name"] = str(row.get(name_col, "")).strip()
                    out_row["Validation_Flags"] = json.dumps(["no_valid_json"], ensure_ascii=False)
                    out_row["Raw_Model_Output"] = model_text_s
                    out_row["Reason"] = "no_valid_json_catastrophic_failure"
                    manual_buffer.append(out_row) # Add to buffer
                    batch_manual_count += 1
                raise Exception("Catastrophic JSON parsing failure.")


            result_map = {}
            if isinstance(parsed_results, list):
                for obj in parsed_results:
                    if isinstance(obj, dict):
                        try:
                            idx = int(obj.get("input_index", 0))
                            result_map[idx] = obj
                        except:
                            logging.warning("Result object missing or malformed input_index: %s", json.dumps(obj))
                            pass
            
            logging.info("JSON successfully parsed. Starting row-by-row validation...")

            for original_idx, row in batch_df.iterrows():
                input_index = original_idx + 1
                name = str(row.get(name_col, "")).strip()
                matched = result_map.get(input_index)

                validation_flags = []
                education_items = []
                summary = ""
                sources_list = []
                
                # --- START ROW PROCESSING ---
                
                if matched is None:
                    validation_flags.append("no_matching_object_in_batch_json")
                    summary = "No verified education info found"
                    logging.info("Row %d (%s) failed to match index in JSON.", input_index, name)
                else:
                    raw_edu = matched.get("education", []) if isinstance(matched, dict) else []
                    summary = matched.get("best_summary", "") or ""
                    sources_list_raw = matched.get("sources", []) or []

                    cleaned_ed, flags = validate_education_items(raw_edu)
                    validation_flags.extend(flags)

                    # --- Source Resolution ---
                    resolved_sources_list = []
                    for s in sources_list_raw:
                        resolved_url, status = resolve_url(s)
                        resolved_sources_list.append(resolved_url)
                        append_row_to_csv(RESOLVED_URLS_FILE, {"original_url": s, "resolved_url": resolved_url, "status_code": status}, ["original_url", "resolved_url", "status_code"])

                    for e in cleaned_ed:
                        s = e.get("source", "")
                        if s:
                            resolved_url, status = resolve_url(s)
                            e["resolved_source"] = resolved_url
                            append_row_to_csv(RESOLVED_URLS_FILE, {"original_url": s, "resolved_url": resolved_url, "status_code": status}, ["original_url", "resolved_url", "status_code"])
                        else:
                            e["resolved_source"] = ""
                            validation_flags.append("education_item_missing_source")
                    # --- END Source Resolution ---

                    education_items = cleaned_ed
                    sources_list = resolved_sources_list

                    if not education_items:
                        summary = "No verified education info found"
                        sources_list = []
                        validation_flags.append("no_education_found")

                    returned_name = matched.get("input_name", "").strip() if isinstance(matched, dict) else ""
                    if returned_name and returned_name.lower() != name.lower():
                        validation_flags.append("returned_name_differs_from_input")
                        
                    if not education_items and name and not summary:
                         validation_flags.append("education_missing_but_summary_empty")
                         logging.warning("Row %d: Education and summary are missing/empty.", input_index)

                # --- Output Row Construction ---
                out_row = row.copy()
                out_row["input_index"] = input_index
                out_row["input_name"] = name
                out_row["Education_JSON"] = json.dumps(education_items, ensure_ascii=False)
                out_row["Education_Summary"] = summary
                out_row["Education_Sources"] = json.dumps(sources_list, ensure_ascii=False)
                out_row["Validation_Flags"] = json.dumps(validation_flags, ensure_ascii=False)
                out_row["Raw_Model_Output"] = model_text_s if (matched is None or validation_flags) else ""

                manual_keywords = {"no_valid_json", "no_matching_object_in_batch_json",
                                 "graduation_year_in_future", "degree_without_source",
                                 "no_resolved_authoritative_source", "education_missing_but_summary_empty"}

                should_manual = any(any(k in v for k in manual_keywords) for v in validation_flags)
                
                destination = "MANUAL REVIEW" if should_manual else "OUTPUT"
                print(f"     Row {input_index} ({name}): Destination: {destination} (Flags: {', '.join(validation_flags) or 'None'})")

                if should_manual:
                    out_row_for_manual = out_row.copy()
                    out_row_for_manual["Reason"] = ";".join(validation_flags) if validation_flags else "other"
                    manual_buffer.append(out_row_for_manual) # Append to buffer
                    logging.info("Row %d (%s) appended to manual review buffer.", input_index, name)
                    batch_manual_count += 1
                else:
                    output_buffer.append(out_row) # Append to buffer
                    logging.info("Row %d (%s) appended to main output buffer.", input_index, name)
                    batch_success_count += 1
            
            
            # --- WRITE BUFFERS TO CSV ---
            if output_buffer:
                df_out = pd.DataFrame(output_buffer)
                # Use to_csv append mode, but without writing headers every time
                df_out.to_csv(OUTPUT_FILE, mode="a", header=False, index=False, columns=csv_columns, encoding="utf-8")
                logging.info("Wrote %d rows to main output file.", len(output_buffer))

            if manual_buffer:
                df_manual = pd.DataFrame(manual_buffer)
                df_manual.to_csv(MANUAL_REVIEW_FILE, mode="a", header=False, index=False, columns=manual_columns, encoding="utf-8")
                logging.info("Wrote %d rows to manual review file.", len(manual_buffer))


            # Print batch summary
            print(f"  -> Batch done. Processed {len(batch_df)} rows. ({batch_success_count} written to OUTPUT, {batch_manual_count} written to MANUAL REVIEW)")

            daily_calls += 1
            if daily_calls < DAILY_LIMIT:
                print(f"  -> Pausing for {SLEEP_SECONDS} seconds before next request...")
                time.sleep(SLEEP_SECONDS)

        except Exception as exc:
            msg = str(exc)
            logging.exception("Error during batch processing for indices %d-%d: %s", batch_start_index, batch_end_index, msg)
            print(f"\n  -> CRITICAL ERROR during batch {batch_start_index}-{batch_end_index}: {msg}")
            
            if "429" in msg or "Resource exhausted" in msg or "RateLimit" in msg:
                delay_match = re.search(r"Please retry in (\d+\.?\d*)s", msg, re.IGNORECASE)
                wait_time = float(delay_match.group(1)) + 1.0 if delay_match else SLEEP_SECONDS * 2
                
                logging.warning("Rate limit detected. Pausing for %s seconds and stopping run.", wait_time)
                print(f"Rate limit detected. Waiting for {wait_time:.1f} seconds. Stopping run to comply.")
                time.sleep(wait_time)
                break
            
            print("Fatal error occurred. Stopping run to prevent further issues.")
            break
        
    logging.info("Run finished")

if __name__ == "__main__":
    main()