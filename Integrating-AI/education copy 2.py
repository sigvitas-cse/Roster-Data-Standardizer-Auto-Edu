#!/usr/bin/env python3
"""
education_extraction_structured.py

Reads input_data.xlsx and queries Gemini (via google.genai client + GoogleSearch tool)
to fetch structured education histories. Saves results to education_output.csv.

Key features:
- Strict JSON output requested from the model (education entries + sources).
- Validates graduation years and flags improbable years.
- Resumes from existing CSV.
- Handles 429 by stopping gracefully so you can re-run next day.
"""

import os
import time
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import tenacity
from google import genai
from google.genai import types

# ------------ CONFIG --------------
INPUT_FILE = "input_data.xlsx"
OUTPUT_FILE = "education_output.csv"
LOG_FILE = "education_extraction_structured.log"

MODEL = "gemini-2.5-flash"   # keep your model choice; change to '-pro' if available/preferred
SLEEP_SECONDS = 4
RETRY_ATTEMPTS = 3
BATCH_SIZE = 1               # keep 1 for maximum accuracy; increase to group multiple names per call
DAILY_LIMIT = 250
# ----------------------------------

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("Missing GEMINI_API_KEY in environment")

client = genai.Client(api_key=api_key)
current_year = datetime.now().year

PROMPT_TEMPLATE = """
You are a careful researcher. For each person provided, find authoritative public sources (law firm bio, university page, company profile, official directory, or LinkedIn) and extract a complete education history.

**INPUT**:
Name: {name}
Organization: {org}
Location: {city}, {state}

**OUTPUT (STRICT JSON ONLY)**:
Return a JSON array (no surrounding text). Each array item must be an object with:
{{
  "input_index": <integer>,
  "input_name": "<original name>",
  "education": [
    {{
      "degree": "<e.g., J.D., LL.M., B.S., M.S., Ph.D.>",
      "field": "<field or major if available, else empty>",
      "institution": "<full name>",
      "graduation_year": "<YYYY or empty string>",
      "notes": "<honors, thesis, or extra details>",
      "source": "<authoritative URL or empty>"
    }},
    ...
  ],
  "best_summary": "<one-sentence human summary>",
  "sources": ["<list of URLs used>"]
}}

Rules:
- If you find nothing, return the object with "education": [] and "sources": [] and best_summary "No education info found".
- Always include at least one source per education item when possible.
- Use exact institution names and degree labels as shown on the sources.
- Do NOT invent degrees or years. If uncertain, leave year blank and include the source.
- Output must be valid JSON only (an array of objects). Number objects using input_index you will supply.
"""

@tenacity.retry(
    stop=tenacity.stop_after_attempt(RETRY_ATTEMPTS),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
    retry=tenacity.retry_if_exception_type(Exception),
    reraise=True
)
def call_model_with_search(prompt_text: str):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt_text,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            response_modalities=["TEXT"]
        )
    )
    text = getattr(response, "text", None)
    if text is None:
        # fallback to string conversion
        text = str(response)
    return text

def extract_json(text: str):
    """Find first JSON array in the text and parse it."""
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1 or end <= start:
        return None
    json_str = text[start:end+1]
    # common cleanup
    json_str = json_str.replace('“', '"').replace('”', '"').replace("’", "'")
    json_str = json_str.replace('\n', ' ')
    json_str = json_str.replace(', }', ' }').replace(', ]', ' ]')
    try:
        return json.loads(json_str)
    except Exception as e:
        logging.exception(f"JSON parse error: {e}")
        return None

def validate_education_entries(ed_list):
    """Validate years and flag improbable years. Return cleaned list and list of flags."""
    flags = []
    cleaned = []
    for entry in ed_list:
        # Ensure required keys exist
        degree = entry.get("degree", "") if isinstance(entry, dict) else ""
        field = entry.get("field", "") if isinstance(entry, dict) else ""
        institution = entry.get("institution", "") if isinstance(entry, dict) else ""
        year_raw = entry.get("graduation_year", "") if isinstance(entry, dict) else ""
        notes = entry.get("notes", "") if isinstance(entry, dict) else ""
        source = entry.get("source", "") if isinstance(entry, dict) else ""

        # normalize year to 4-digit if possible
        year = ""
        if isinstance(year_raw, int):
            year = str(year_raw)
        elif isinstance(year_raw, str) and year_raw.strip().isdigit():
            year = year_raw.strip()
        elif isinstance(year_raw, str):
            # try to extract 4-digit number
            import re
            m = re.search(r'(19|20)\d{2}', year_raw)
            if m:
                year = m.group(0)

        # flag improbable years
        if year:
            try:
                y = int(year)
                if y > current_year:
                    flags.append(f"graduation_year_in_future:{year} for institution {institution}")
                elif y < 1900:
                    flags.append(f"graduation_year_too_old:{year} for institution {institution}")
            except Exception:
                pass

        cleaned.append({
            "degree": degree or "",
            "field": field or "",
            "institution": institution or "",
            "graduation_year": year,
            "notes": notes or "",
            "source": source or ""
        })
    return cleaned, flags

def find_name_column(df: pd.DataFrame):
    # prefer exact 'Name', fallback to any header containing 'name'
    if "Name" in df.columns:
        return "Name"
    for c in df.columns:
        if "name" in c.lower():
            return c
    return df.columns[0]

def main():
    logging.info("Starting structured education extraction run")
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"Input file not found: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE, engine="openpyxl")
    name_col = find_name_column(df)
    total_rows = len(df)
    logging.info(f"Loaded {total_rows} rows. Using name column '{name_col}'")

    # prepare output CSV (resume if exists)
    processed_names = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            if "input_name" in existing.columns:
                processed_names = set(existing["input_name"].astype(str).tolist())
            logging.info(f"Resuming; found {len(processed_names)} already-processed names")
            out_handle = open(OUTPUT_FILE, "a", encoding="utf-8", newline="")
            append_header = False
        except Exception:
            # fallback: create new
            processed_names = set()
            append_header = True
            out_handle = open(OUTPUT_FILE, "w", encoding="utf-8", newline="")
    else:
        out_handle = open(OUTPUT_FILE, "w", encoding="utf-8", newline="")
        append_header = True

    cols_for_csv = list(df.columns) + ["input_index", "input_name", "Education_JSON", "Education_Summary", "Education_Sources", "Validation_Flags", "Raw_Model_Output"]
    # write header if needed
    if append_header:
        pd.DataFrame(columns=cols_for_csv).to_csv(out_handle, index=False)
    out_handle.close()

    daily_calls = 0
    results = []
    for i, row in df.iterrows():
        if daily_calls >= DAILY_LIMIT:
            logging.warning("Reached daily API call limit; stopping run")
            print(f"Reached daily limit ({DAILY_LIMIT}). Stopping.")
            break

        raw_name = row.get(name_col, "")
        if pd.isna(raw_name) or str(raw_name).strip() == "":
            continue
        name = str(raw_name).strip()
        if name in processed_names:
            logging.info(f"Skipping {name} (already done)")
            continue

        org = row.get("Organization/Law Firm Name", "")
        city = row.get("City", "")
        state = row.get("State", "")

        input_index = i + 1
        prompt = PROMPT_TEMPLATE.format(name=name, org=org or "", city=city or "", state=state or "")

        print(f"[{input_index}/{total_rows}] Querying: {name}")
        try:
            model_text = call_model_with_search(prompt)
            parsed = extract_json(model_text)
            validation_flags = []
            education_json = []
            summary = ""
            sources_list = []

            if parsed is None:
                # fallback: save raw model text for inspection
                logging.warning(f"No JSON returned for {name}. Saving raw text.")
                education_json = []
                summary = ""
                sources_list = []
                validation_flags.append("no_valid_json")
            else:
                # parsed is expected to be an array; find object matching input_index
                matched_obj = None
                if isinstance(parsed, list):
                    # try match by input_index first
                    for obj in parsed:
                        try:
                            if int(obj.get("input_index", -1)) == input_index:
                                matched_obj = obj
                                break
                        except Exception:
                            pass
                    # fallback: match by name (case-insensitive)
                    if matched_obj is None:
                        for obj in parsed:
                            if isinstance(obj, dict) and obj.get("input_name", "").strip().lower() == name.lower():
                                matched_obj = obj
                                break
                    # fallback: if only one object returned for this batch, use it
                    if matched_obj is None and len(parsed) == 1:
                        matched_obj = parsed[0]

                if matched_obj is None:
                    logging.warning(f"No matching object for {name} in model JSON")
                    validation_flags.append("no_matching_object_in_json")
                    education_json = []
                    summary = ""
                    sources_list = []
                else:
                    raw_ed = matched_obj.get("education", [])
                    cleaned_ed, flags = validate_education_entries(raw_ed)
                    education_json = cleaned_ed
                    validation_flags.extend(flags)
                    summary = matched_obj.get("best_summary", "") or ""
                    sources_list = matched_obj.get("sources", []) or []

            # prepare row to append
            out_row = row.copy()
            out_row["input_index"] = input_index
            out_row["input_name"] = name
            out_row["Education_JSON"] = json.dumps(education_json, ensure_ascii=False)
            out_row["Education_Summary"] = summary
            out_row["Education_Sources"] = json.dumps(sources_list, ensure_ascii=False)
            out_row["Validation_Flags"] = json.dumps(validation_flags, ensure_ascii=False)
            out_row["Raw_Model_Output"] = model_text if model_text else ""

            # append to CSV incrementally
            pd.DataFrame([out_row]).to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
            processed_names.add(name)
            daily_calls += 1
            time.sleep(SLEEP_SECONDS)
            print(f"  -> Saved (flags: {validation_flags})")

        except Exception as exc:
            msg = str(exc)
            logging.exception(f"Error querying {name}: {msg}")
            print(f"  -> Error: {msg}")
            # if rate-limit style error, stop so you can resume next day
            if "429" in msg or "Resource exhausted" in msg or "RateLimit" in msg:
                print("Rate limit / resource exhausted detected. Exiting to allow resume later.")
                break
            # otherwise save an error row
            out_row = row.copy()
            out_row["input_index"] = input_index
            out_row["input_name"] = name
            out_row["Education_JSON"] = json.dumps([], ensure_ascii=False)
            out_row["Education_Summary"] = ""
            out_row["Education_Sources"] = json.dumps([], ensure_ascii=False)
            out_row["Validation_Flags"] = json.dumps([f"exception:{msg}"], ensure_ascii=False)
            out_row["Raw_Model_Output"] = f"ERROR: {msg}"
            pd.DataFrame([out_row]).to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
            # continue to next row (or break if you prefer)

    logging.info("Extraction run finished")

if __name__ == "__main__":
    main()
