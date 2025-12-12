#!/usr/bin/env python3
"""
education_extraction_final.py

Robust structured education extraction for people in input_data.xlsx using google.genai
with GoogleSearch tool. Resolves redirect URLs, validates results, and writes a manual-review CSV.

- Test with TEST_COUNT rows before running all data.
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
LOG_FILE = "education_extraction_final.log"

MODEL = "gemini-2.5-flash"
SLEEP_SECONDS = 4
RETRY_ATTEMPTS = 3
DAILY_LIMIT = 250
BATCH_SIZE = 1        # keep 1 for best mapping correctness; increase at your own risk
TEST_COUNT = 100      # change to None to run all rows
# ----------------------------------------

# Logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("Missing GEMINI_API_KEY in environment")

client = genai.Client(api_key=api_key)
current_year = datetime.now().year

# Strict prompt (improved)
PROMPT_TEMPLATE = """
You are a careful, literal data researcher. For the input person below, find authoritative public sources
(law firm bio, company bio, university profile, LinkedIn profile, or official directories) and extract
a complete education history.

**IMPORTANT RULES (read and obey):**
1. Use EXACT input_index provided. **Do not renumber** results. Set "input_index" in the output equal to the input_index you were given.
2. Return STRICT JSON ONLY (no commentary) — a JSON array of objects (one object per input). Each object must be:
{{
  "input_index": <integer>,
  "input_name": "<original name>",
  "education": [
    {{
      "degree": "<e.g., J.D., LL.M., B.S., M.S., Ph.D.>",
      "field": "<field/major or empty>",
      "institution": "<full institution name>",
      "graduation_year": "<YYYY or empty>",
      "notes": "<honors/in progress/other or empty>",
      "source": "<canonical URL to the authoritative source (NOT an internal redirect) or empty>"
    }},
    ...
  ],
  "best_summary": "<one-sentence summary or 'No verified education info found'>",
  "sources": ["<canonical URLs used>"]
}}

3. **Do NOT invent degrees or years.** If a degree is in progress, set graduation_year to "" and notes to "in progress/expected - cite source".
4. Return canonical page URLs (the real page URL, not an internal redirect). If only a redirect is available, return both in `source` separated by a space, but prefer canonical URL.
5. If you cannot find authoritative sources, set "education": [], "sources": [], and "best_summary": "No verified education info found".
6. Output must be valid JSON (an array). Use the exact input_index integer provided.
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
    # remove code fences ``` or ```json
    if t.startswith("```"):
        # remove leading fence block
        parts = t.split("\n")
        # drop the first line and last fence if present
        if len(parts) >= 2:
            # drop first line
            parts = parts[1:]
            # if last line is a fence, drop it
            if parts and parts[-1].strip().startswith("```"):
                parts = parts[:-1]
            t = "\n".join(parts).strip()
    # unescape double-escaped quotes if present
    t = t.replace('\\"', '"')
    # normalize smart quotes
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
    # cleanup common trailing commas
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
        # sometimes model returns both (redirect canonical) space separated; take first token
        cleaned = str(url).strip()
        # If model returned "redirect canonical", try to split and pick canonical-looking one
        # We'll attempt the last token if it looks like http
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
        grad_year_raw = ent.get("graduation_year", "") or ""
        notes = ent.get("notes", "") or ""
        source = ent.get("source", "") or ""
        # normalize year
        year = ""
        if isinstance(grad_year_raw, int):
            year = str(grad_year_raw)
        elif isinstance(grad_year_raw, str):
            m = re.search(r'(19|20)\d{2}', grad_year_raw)
            if m:
                year = m.group(0)
        # flags
        if year:
            try:
                y = int(year)
                if y > current_year:
                    flags.append(f"graduation_year_in_future:{year} for {institution}")
            except Exception:
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
    df = pd.DataFrame([row_dict])
    df.to_csv(path, mode="a", header=False, index=False, columns=columns, encoding="utf-8")

def initialize_csv_if_needed(path, columns):
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8")

def main():
    logging.info("Starting extraction run")
    if not os.path.exists(INPUT_FILE):
        logging.error("Input file missing: %s", INPUT_FILE)
        raise SystemExit("Input file not found")

    df = pd.read_excel(INPUT_FILE, engine="openpyxl")
    name_col = find_name_column(df)
    total = len(df)
    logging.info("Loaded %d rows; using name column: %s", total, name_col)

    # If TEST_COUNT is set, limit the DataFrame
    if TEST_COUNT and isinstance(TEST_COUNT, int):
        df = df.head(TEST_COUNT)
        logging.info("TEST_COUNT set: running first %d rows only", TEST_COUNT)

    # Prepare output files
    csv_columns = list(df.columns) + ["input_index", "input_name", "Education_JSON", "Education_Summary", "Education_Sources", "Validation_Flags", "Raw_Model_Output"]
    initialize_csv_if_needed(OUTPUT_FILE, csv_columns)
    manual_columns = csv_columns + ["Reason"]
    initialize_csv_if_needed(MANUAL_REVIEW_FILE, manual_columns)
    initialize_csv_if_needed(RESOLVED_URLS_FILE, ["original_url", "resolved_url", "status_code"])

    # load processed names to resume
    try:
        processed_df = pd.read_csv(OUTPUT_FILE, encoding="utf-8")
        processed_names = set(processed_df.get("input_name", []).astype(str).tolist())
        logging.info("Resuming; found %d processed names", len(processed_names))
    except Exception:
        processed_names = set()

    daily_calls = 0

    for idx, row in df.iterrows():
        if daily_calls >= DAILY_LIMIT:
            logging.warning("Reached daily limit; stopping")
            print(f"Reached daily limit ({DAILY_LIMIT}). Stop and resume later.")
            break

        raw_name = row.get(name_col, "")
        if pd.isna(raw_name) or str(raw_name).strip() == "":
            continue
        name = str(raw_name).strip()
        if name in processed_names:
            logging.info("Skipping already processed: %s", name)
            continue

        org = row.get("Organization/Law Firm Name", "") or ""
        city = row.get("City", "") or ""
        state = row.get("State", "") or ""
        input_index = int(idx) + 1  # 1-based index

        # Build prompt for single person
        prompt = PROMPT_TEMPLATE + f"\n\nINPUT_INDEX: {input_index}\nName: {name}\nOrganization: {org}\nLocation: {city}, {state}\n\nReturn JSON now."
        print(f"[{input_index}] Querying: {name} ...")
        try:
            model_text = call_model(prompt)
            model_text_s = sanitize_model_text(model_text)
            parsed = extract_json_array(model_text_s)

            validation_flags = []
            education_items = []
            summary = ""
            sources_list = []
            raw_output_saved = model_text_s

            if parsed is None:
                # No valid JSON returned
                logging.warning("No valid JSON for %s (index %d)", name, input_index)
                validation_flags.append("no_valid_json")
                education_items = []
                summary = ""
                sources_list = []
                # Save raw output to manual review
                manual_row = row.copy()
                manual_row["input_index"] = input_index
                manual_row["input_name"] = name
                manual_row["Education_JSON"] = json.dumps([], ensure_ascii=False)
                manual_row["Education_Summary"] = ""
                manual_row["Education_Sources"] = json.dumps([], ensure_ascii=False)
                manual_row["Validation_Flags"] = json.dumps(validation_flags, ensure_ascii=False)
                manual_row["Raw_Model_Output"] = model_text_s
                manual_row["Reason"] = "no_valid_json"
                append_row_to_csv(MANUAL_REVIEW_FILE, manual_row, manual_columns)
            else:
                # parsed should be list; find matching object by input_index first
                matched = None
                if isinstance(parsed, list):
                    for obj in parsed:
                        try:
                            if int(obj.get("input_index", -999999)) == input_index:
                                matched = obj
                                break
                        except Exception:
                            pass
                    # fallback match by name
                    if matched is None:
                        for obj in parsed:
                            if isinstance(obj, dict) and obj.get("input_name", "").strip().lower() == name.lower():
                                matched = obj
                                break
                    # fallback: if single object returned, take it but flag
                    if matched is None and len(parsed) == 1:
                        matched = parsed[0]
                        validation_flags.append("used_single_response_for_batch")
                else:
                    validation_flags.append("parsed_json_not_list")

                if matched is None:
                    # no match found -> manual review
                    logging.warning("No matching object in JSON for %s (idx %d)", name, input_index)
                    validation_flags.append("no_matching_object_in_json")
                    manual_row = row.copy()
                    manual_row["input_index"] = input_index
                    manual_row["input_name"] = name
                    manual_row["Education_JSON"] = json.dumps(parsed, ensure_ascii=False)
                    manual_row["Education_Summary"] = ""
                    manual_row["Education_Sources"] = json.dumps([], ensure_ascii=False)
                    manual_row["Validation_Flags"] = json.dumps(validation_flags, ensure_ascii=False)
                    manual_row["Raw_Model_Output"] = model_text_s
                    manual_row["Reason"] = "no_matching_object_in_json"
                    append_row_to_csv(MANUAL_REVIEW_FILE, manual_row, manual_columns)
                else:
                    # Extract fields
                    raw_edu = matched.get("education", []) if isinstance(matched, dict) else []
                    summary = matched.get("best_summary", "") or ""
                    sources_list = matched.get("sources", []) or []
                    # Validate education list
                    cleaned_ed, flags = validate_education_items(raw_edu)
                    validation_flags.extend(flags)

                    # Resolve sources: for each education item source and for global sources_list
                    resolved_sources = []
                    for s in sources_list:
                        resolved_url, status = resolve_url(s)
                        resolved_sources.append(resolved_url)
                        # log mapping
                        append_row_to_csv(RESOLVED_URLS_FILE, {"original_url": s, "resolved_url": resolved_url, "status_code": status}, ["original_url", "resolved_url", "status_code"])

                    # For each education entry, try to resolve its source
                    for e in cleaned_ed:
                        s = e.get("source", "")
                        if s:
                            resolved_url, status = resolve_url(s)
                            e["resolved_source"] = resolved_url
                            append_row_to_csv(RESOLVED_URLS_FILE, {"original_url": s, "resolved_url": resolved_url, "status_code": status}, ["original_url", "resolved_url", "status_code"])
                        else:
                            e["resolved_source"] = ""
                            validation_flags.append("education_item_missing_source")

                    # If any education item has no source -> mark unverified
                    unverified = any((not e.get("source") and not e.get("resolved_source")) for e in cleaned_ed)
                    if unverified and cleaned_ed:
                        validation_flags.append("some_edu_items_unverified")

                    # If any future-year flag exists -> mark for manual review
                    future_flags = [f for f in validation_flags if f.startswith("graduation_year_in_future")]
                    if future_flags:
                        # Keep the entry but flag for review
                        validation_flags.extend(["needs_manual_review_future_year"])

                    education_items = cleaned_ed
                    # combine resolved sources
                    sources_list = resolved_sources

                    # If no education found -> set summary to "No verified education info found"
                    if not education_items:
                        summary = "No verified education info found"
                        sources_list = []
                        validation_flags.append("no_education_found")

                    # If education exists but no authoritative resolved sources found -> manual review
                    if education_items and not any(e.get("resolved_source") for e in education_items):
                        validation_flags.append("no_resolved_authoritative_source")

                    # If matched input_name doesn't match provided name exactly -> flag
                    returned_name = matched.get("input_name", "").strip() if isinstance(matched, dict) else ""
                    if returned_name and returned_name.lower() != name.lower():
                        validation_flags.append("returned_name_differs_from_input")

            # Build final out row
            out_row = row.copy()
            out_row["input_index"] = input_index
            out_row["input_name"] = name
            out_row["Education_JSON"] = json.dumps(education_items, ensure_ascii=False)
            out_row["Education_Summary"] = summary
            out_row["Education_Sources"] = json.dumps(sources_list, ensure_ascii=False)
            out_row["Validation_Flags"] = json.dumps(validation_flags, ensure_ascii=False)
            out_row["Raw_Model_Output"] = raw_output_saved or ""

            # If there are severe flags, write to manual review instead of main output
            severe_flags = [f for f in validation_flags if f in ("no_valid_json", "no_matching_object_in_json", "graduation_year_in_future:"+str(current_year+1))]  # example
            # better logic: any of these flags should likely be manually reviewed
            manual_keywords = {"no_valid_json", "no_matching_object_in_json", "graduation_year_in_future", "degree_without_source", "some_edu_items_unverified", "no_resolved_authoritative_source"}
            should_manual = any(any(k in v for k in manual_keywords) for v in validation_flags)

            if should_manual:
                # append to manual review CSV
                out_row_for_manual = out_row.copy()
                out_row_for_manual["Reason"] = ";".join(validation_flags) if validation_flags else "other"
                append_row_to_csv(MANUAL_REVIEW_FILE, out_row_for_manual, manual_columns)
                logging.info("Appended %s to manual review (flags: %s)", name, validation_flags)
            else:
                # append to main CSV
                append_row_to_csv(OUTPUT_FILE, out_row, csv_columns)
                logging.info("Appended %s to output (flags: %s)", name, validation_flags)
                processed_names.add(name)

            daily_calls += 1
            # brief console message
            print(f"  -> Done (flags: {validation_flags})")
            time.sleep(SLEEP_SECONDS)

        except Exception as exc:
            msg = str(exc)
            logging.exception("Error for %s: %s", name, msg)
            print(f"  -> Error: {msg}")
            # If rate-limit-like message, stop and allow resume
            if "429" in msg or "Resource exhausted" in msg or "RateLimit" in msg:
                print("Rate limit detected; stopping to allow resume later.")
                break
            # Otherwise, append to manual review with error
            manual_row = row.copy()
            manual_row["input_index"] = input_index
            manual_row["input_name"] = name
            manual_row["Education_JSON"] = json.dumps([], ensure_ascii=False)
            manual_row["Education_Summary"] = ""
            manual_row["Education_Sources"] = json.dumps([], ensure_ascii=False)
            manual_row["Validation_Flags"] = json.dumps([f"exception:{msg}"], ensure_ascii=False)
            manual_row["Raw_Model_Output"] = f"ERROR: {msg}"
            manual_row["Reason"] = "exception"
            append_row_to_csv(MANUAL_REVIEW_FILE, manual_row, manual_columns)
            # continue to next row

    logging.info("Run finished")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
education_extraction_final.py

Robust structured education extraction for people in input_data.xlsx using google.genai
with GoogleSearch tool. Resolves redirect URLs, validates results, and writes a manual-review CSV.

- Test with TEST_COUNT rows before running all data.
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
LOG_FILE = "education_extraction_final.log"

MODEL = "gemini-2.5-flash"
SLEEP_SECONDS = 4
RETRY_ATTEMPTS = 3
DAILY_LIMIT = 250
BATCH_SIZE = 1        # keep 1 for best mapping correctness; increase at your own risk
TEST_COUNT = 100      # change to None to run all rows
# ----------------------------------------

# Logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("Missing GEMINI_API_KEY in environment")

client = genai.Client(api_key=api_key)
current_year = datetime.now().year

# Strict prompt (improved)
PROMPT_TEMPLATE = """
You are a careful, literal data researcher. For the input person below, find authoritative public sources
(law firm bio, company bio, university profile, LinkedIn profile, or official directories) and extract
a complete education history.

**IMPORTANT RULES (read and obey):**
1. Use EXACT input_index provided. **Do not renumber** results. Set "input_index" in the output equal to the input_index you were given.
2. Return STRICT JSON ONLY (no commentary) — a JSON array of objects (one object per input). Each object must be:
{{
  "input_index": <integer>,
  "input_name": "<original name>",
  "education": [
    {{
      "degree": "<e.g., J.D., LL.M., B.S., M.S., Ph.D.>",
      "field": "<field/major or empty>",
      "institution": "<full institution name>",
      "graduation_year": "<YYYY or empty>",
      "notes": "<honors/in progress/other or empty>",
      "source": "<canonical URL to the authoritative source (NOT an internal redirect) or empty>"
    }},
    ...
  ],
  "best_summary": "<one-sentence summary or 'No verified education info found'>",
  "sources": ["<canonical URLs used>"]
}}

3. **Do NOT invent degrees or years.** If a degree is in progress, set graduation_year to "" and notes to "in progress/expected - cite source".
4. Return canonical page URLs (the real page URL, not an internal redirect). If only a redirect is available, return both in `source` separated by a space, but prefer canonical URL.
5. If you cannot find authoritative sources, set "education": [], "sources": [], and "best_summary": "No verified education info found".
6. Output must be valid JSON (an array). Use the exact input_index integer provided.
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
    # remove code fences ``` or ```json
    if t.startswith("```"):
        # remove leading fence block
        parts = t.split("\n")
        # drop the first line and last fence if present
        if len(parts) >= 2:
            # drop first line
            parts = parts[1:]
            # if last line is a fence, drop it
            if parts and parts[-1].strip().startswith("```"):
                parts = parts[:-1]
            t = "\n".join(parts).strip()
    # unescape double-escaped quotes if present
    t = t.replace('\\"', '"')
    # normalize smart quotes
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
    # cleanup common trailing commas
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
        # sometimes model returns both (redirect canonical) space separated; take first token
        cleaned = str(url).strip()
        # If model returned "redirect canonical", try to split and pick canonical-looking one
        # We'll attempt the last token if it looks like http
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
        grad_year_raw = ent.get("graduation_year", "") or ""
        notes = ent.get("notes", "") or ""
        source = ent.get("source", "") or ""
        # normalize year
        year = ""
        if isinstance(grad_year_raw, int):
            year = str(grad_year_raw)
        elif isinstance(grad_year_raw, str):
            m = re.search(r'(19|20)\d{2}', grad_year_raw)
            if m:
                year = m.group(0)
        # flags
        if year:
            try:
                y = int(year)
                if y > current_year:
                    flags.append(f"graduation_year_in_future:{year} for {institution}")
            except Exception:
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
    df = pd.DataFrame([row_dict])
    df.to_csv(path, mode="a", header=False, index=False, columns=columns, encoding="utf-8")

def initialize_csv_if_needed(path, columns):
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8")

def main():
    logging.info("Starting extraction run")
    if not os.path.exists(INPUT_FILE):
        logging.error("Input file missing: %s", INPUT_FILE)
        raise SystemExit("Input file not found")

    df = pd.read_excel(INPUT_FILE, engine="openpyxl")
    name_col = find_name_column(df)
    total = len(df)
    logging.info("Loaded %d rows; using name column: %s", total, name_col)

    # If TEST_COUNT is set, limit the DataFrame
    if TEST_COUNT and isinstance(TEST_COUNT, int):
        df = df.head(TEST_COUNT)
        logging.info("TEST_COUNT set: running first %d rows only", TEST_COUNT)

    # Prepare output files
    csv_columns = list(df.columns) + ["input_index", "input_name", "Education_JSON", "Education_Summary", "Education_Sources", "Validation_Flags", "Raw_Model_Output"]
    initialize_csv_if_needed(OUTPUT_FILE, csv_columns)
    manual_columns = csv_columns + ["Reason"]
    initialize_csv_if_needed(MANUAL_REVIEW_FILE, manual_columns)
    initialize_csv_if_needed(RESOLVED_URLS_FILE, ["original_url", "resolved_url", "status_code"])

    # load processed names to resume
    try:
        processed_df = pd.read_csv(OUTPUT_FILE, encoding="utf-8")
        processed_names = set(processed_df.get("input_name", []).astype(str).tolist())
        logging.info("Resuming; found %d processed names", len(processed_names))
    except Exception:
        processed_names = set()

    daily_calls = 0

    for idx, row in df.iterrows():
        if daily_calls >= DAILY_LIMIT:
            logging.warning("Reached daily limit; stopping")
            print(f"Reached daily limit ({DAILY_LIMIT}). Stop and resume later.")
            break

        raw_name = row.get(name_col, "")
        if pd.isna(raw_name) or str(raw_name).strip() == "":
            continue
        name = str(raw_name).strip()
        if name in processed_names:
            logging.info("Skipping already processed: %s", name)
            continue

        org = row.get("Organization/Law Firm Name", "") or ""
        city = row.get("City", "") or ""
        state = row.get("State", "") or ""
        input_index = int(idx) + 1  # 1-based index

        # Build prompt for single person
        prompt = PROMPT_TEMPLATE + f"\n\nINPUT_INDEX: {input_index}\nName: {name}\nOrganization: {org}\nLocation: {city}, {state}\n\nReturn JSON now."
        print(f"[{input_index}] Querying: {name} ...")
        try:
            model_text = call_model(prompt)
            model_text_s = sanitize_model_text(model_text)
            parsed = extract_json_array(model_text_s)

            validation_flags = []
            education_items = []
            summary = ""
            sources_list = []
            raw_output_saved = model_text_s

            if parsed is None:
                # No valid JSON returned
                logging.warning("No valid JSON for %s (index %d)", name, input_index)
                validation_flags.append("no_valid_json")
                education_items = []
                summary = ""
                sources_list = []
                # Save raw output to manual review
                manual_row = row.copy()
                manual_row["input_index"] = input_index
                manual_row["input_name"] = name
                manual_row["Education_JSON"] = json.dumps([], ensure_ascii=False)
                manual_row["Education_Summary"] = ""
                manual_row["Education_Sources"] = json.dumps([], ensure_ascii=False)
                manual_row["Validation_Flags"] = json.dumps(validation_flags, ensure_ascii=False)
                manual_row["Raw_Model_Output"] = model_text_s
                manual_row["Reason"] = "no_valid_json"
                append_row_to_csv(MANUAL_REVIEW_FILE, manual_row, manual_columns)
            else:
                # parsed should be list; find matching object by input_index first
                matched = None
                if isinstance(parsed, list):
                    for obj in parsed:
                        try:
                            if int(obj.get("input_index", -999999)) == input_index:
                                matched = obj
                                break
                        except Exception:
                            pass
                    # fallback match by name
                    if matched is None:
                        for obj in parsed:
                            if isinstance(obj, dict) and obj.get("input_name", "").strip().lower() == name.lower():
                                matched = obj
                                break
                    # fallback: if single object returned, take it but flag
                    if matched is None and len(parsed) == 1:
                        matched = parsed[0]
                        validation_flags.append("used_single_response_for_batch")
                else:
                    validation_flags.append("parsed_json_not_list")

                if matched is None:
                    # no match found -> manual review
                    logging.warning("No matching object in JSON for %s (idx %d)", name, input_index)
                    validation_flags.append("no_matching_object_in_json")
                    manual_row = row.copy()
                    manual_row["input_index"] = input_index
                    manual_row["input_name"] = name
                    manual_row["Education_JSON"] = json.dumps(parsed, ensure_ascii=False)
                    manual_row["Education_Summary"] = ""
                    manual_row["Education_Sources"] = json.dumps([], ensure_ascii=False)
                    manual_row["Validation_Flags"] = json.dumps(validation_flags, ensure_ascii=False)
                    manual_row["Raw_Model_Output"] = model_text_s
                    manual_row["Reason"] = "no_matching_object_in_json"
                    append_row_to_csv(MANUAL_REVIEW_FILE, manual_row, manual_columns)
                else:
                    # Extract fields
                    raw_edu = matched.get("education", []) if isinstance(matched, dict) else []
                    summary = matched.get("best_summary", "") or ""
                    sources_list = matched.get("sources", []) or []
                    # Validate education list
                    cleaned_ed, flags = validate_education_items(raw_edu)
                    validation_flags.extend(flags)

                    # Resolve sources: for each education item source and for global sources_list
                    resolved_sources = []
                    for s in sources_list:
                        resolved_url, status = resolve_url(s)
                        resolved_sources.append(resolved_url)
                        # log mapping
                        append_row_to_csv(RESOLVED_URLS_FILE, {"original_url": s, "resolved_url": resolved_url, "status_code": status}, ["original_url", "resolved_url", "status_code"])

                    # For each education entry, try to resolve its source
                    for e in cleaned_ed:
                        s = e.get("source", "")
                        if s:
                            resolved_url, status = resolve_url(s)
                            e["resolved_source"] = resolved_url
                            append_row_to_csv(RESOLVED_URLS_FILE, {"original_url": s, "resolved_url": resolved_url, "status_code": status}, ["original_url", "resolved_url", "status_code"])
                        else:
                            e["resolved_source"] = ""
                            validation_flags.append("education_item_missing_source")

                    # If any education item has no source -> mark unverified
                    unverified = any((not e.get("source") and not e.get("resolved_source")) for e in cleaned_ed)
                    if unverified and cleaned_ed:
                        validation_flags.append("some_edu_items_unverified")

                    # If any future-year flag exists -> mark for manual review
                    future_flags = [f for f in validation_flags if f.startswith("graduation_year_in_future")]
                    if future_flags:
                        # Keep the entry but flag for review
                        validation_flags.extend(["needs_manual_review_future_year"])

                    education_items = cleaned_ed
                    # combine resolved sources
                    sources_list = resolved_sources

                    # If no education found -> set summary to "No verified education info found"
                    if not education_items:
                        summary = "No verified education info found"
                        sources_list = []
                        validation_flags.append("no_education_found")

                    # If education exists but no authoritative resolved sources found -> manual review
                    if education_items and not any(e.get("resolved_source") for e in education_items):
                        validation_flags.append("no_resolved_authoritative_source")

                    # If matched input_name doesn't match provided name exactly -> flag
                    returned_name = matched.get("input_name", "").strip() if isinstance(matched, dict) else ""
                    if returned_name and returned_name.lower() != name.lower():
                        validation_flags.append("returned_name_differs_from_input")

            # Build final out row
            out_row = row.copy()
            out_row["input_index"] = input_index
            out_row["input_name"] = name
            out_row["Education_JSON"] = json.dumps(education_items, ensure_ascii=False)
            out_row["Education_Summary"] = summary
            out_row["Education_Sources"] = json.dumps(sources_list, ensure_ascii=False)
            out_row["Validation_Flags"] = json.dumps(validation_flags, ensure_ascii=False)
            out_row["Raw_Model_Output"] = raw_output_saved or ""

            # If there are severe flags, write to manual review instead of main output
            severe_flags = [f for f in validation_flags if f in ("no_valid_json", "no_matching_object_in_json", "graduation_year_in_future:"+str(current_year+1))]  # example
            # better logic: any of these flags should likely be manually reviewed
            manual_keywords = {"no_valid_json", "no_matching_object_in_json", "graduation_year_in_future", "degree_without_source", "some_edu_items_unverified", "no_resolved_authoritative_source"}
            should_manual = any(any(k in v for k in manual_keywords) for v in validation_flags)

            if should_manual:
                # append to manual review CSV
                out_row_for_manual = out_row.copy()
                out_row_for_manual["Reason"] = ";".join(validation_flags) if validation_flags else "other"
                append_row_to_csv(MANUAL_REVIEW_FILE, out_row_for_manual, manual_columns)
                logging.info("Appended %s to manual review (flags: %s)", name, validation_flags)
            else:
                # append to main CSV
                append_row_to_csv(OUTPUT_FILE, out_row, csv_columns)
                logging.info("Appended %s to output (flags: %s)", name, validation_flags)
                processed_names.add(name)

            daily_calls += 1
            # brief console message
            print(f"  -> Done (flags: {validation_flags})")
            time.sleep(SLEEP_SECONDS)

        except Exception as exc:
            msg = str(exc)
            logging.exception("Error for %s: %s", name, msg)
            print(f"  -> Error: {msg}")
            # If rate-limit-like message, stop and allow resume
            if "429" in msg or "Resource exhausted" in msg or "RateLimit" in msg:
                print("Rate limit detected; stopping to allow resume later.")
                break
            # Otherwise, append to manual review with error
            manual_row = row.copy()
            manual_row["input_index"] = input_index
            manual_row["input_name"] = name
            manual_row["Education_JSON"] = json.dumps([], ensure_ascii=False)
            manual_row["Education_Summary"] = ""
            manual_row["Education_Sources"] = json.dumps([], ensure_ascii=False)
            manual_row["Validation_Flags"] = json.dumps([f"exception:{msg}"], ensure_ascii=False)
            manual_row["Raw_Model_Output"] = f"ERROR: {msg}"
            manual_row["Reason"] = "exception"
            append_row_to_csv(MANUAL_REVIEW_FILE, manual_row, manual_columns)
            # continue to next row

    logging.info("Run finished")

if __name__ == "__main__":
    main()
