import pandas as pd
import json

# ==== 1. Load your CSV file ====
# 🔁 Change this to your actual CSV filename
INPUT_FILE = "education_output.csv"

# If you get encoding issues, try encoding="utf-8-sig"
df = pd.read_csv(INPUT_FILE)

print("Rows loaded:", len(df))


# ==== 2. Helper functions to parse JSON safely ====

def parse_education_json(cell):
    """
    Parse the Education_JSON column into a Python list.
    Returns [] if empty/invalid.
    """
    if pd.isna(cell):
        return []
    text = str(cell).strip()
    if text == "" or text == "[]":
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If something is wrong with JSON format
        return []


def parse_from_raw_model(raw_cell):
    """
    Fallback: parse education from Raw_Model_Output, if needed.
    Raw_Model_Output looks like:
    [
      {
        "input_index": 1,
        "input_name": "...",
        "education": [ ... ]
      }
    ]
    """
    if pd.isna(raw_cell):
        return []
    text = str(raw_cell).strip()
    if text == "" or text == "[]":
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return first.get("education", []) or []
    return []


# ==== 3. Build a clean education_list column ====

def merge_education(row):
    """
    Use Education_JSON if present,
    otherwise try Raw_Model_Output.
    """
    edu = parse_education_json(row.get("Education_JSON"))
    if edu:
        return edu
    return parse_from_raw_model(row.get("Raw_Model_Output"))


df["education_list"] = df.apply(merge_education, axis=1)

print("Example parsed education_list for first row:")
print(df["education_list"].iloc[0])


# ==== 4A. NORMALIZED TABLE: one row per degree ====

# Explode so that each education entry becomes one row
df_exploded = df.explode("education_list", ignore_index=True)

# Convert each dict into separate columns
edu_normalized = pd.json_normalize(df_exploded["education_list"])

# Columns from your original CSV that you want to keep
base_cols = [
    "Name",
    "Organization/Law Firm Name",
    "Address Line 1",
    "Address Line 2",
    "City",
    "State",
    "Country",
    "Zipcode",
    "Phone Number",
    "Reg Code",
    "Agent/Attorney",
    "input_index",
    "input_name",
]

result_degrees = pd.concat(
    [df_exploded[base_cols], edu_normalized],
    axis=1
)

# Reorder final columns
result_degrees = result_degrees[
    base_cols
    + ["degree", "field", "institution", "graduation_year", "notes", "source", "resolved_source"]
]

# Save to CSV
result_degrees.to_csv("education_degrees_normalized.csv", index=False)
print("Saved: education_degrees_normalized.csv")


# ==== 4B. FORMATTED TEXT COLUMN PER PERSON ====

def format_education(entries):
    """
    Build a human-readable string from the list of education entries.
    Example:
    "B.S.M.E., Engineering and Applied Sciences, Arizona State University (1987); J.D., Pepperdine University (1995)"
    """
    if not entries:
        return "No verified education info found"

    parts = []
    for e in entries:
        if not isinstance(e, dict):
            continue

        degree = e.get("degree", "")
        field = e.get("field", "")
        institution = e.get("institution", "")
        year = e.get("graduation_year", "")
        notes = e.get("notes", "")

        piece = degree or ""
        if field:
            piece += (", " if piece else "") + field
        if institution:
            piece += (", " if piece else "") + institution
        if year:
            piece += f" ({year})"
        if notes:
            piece += (", " if piece else "") + notes

        if piece:
            parts.append(piece)

    return "; ".join(parts) if parts else "No verified education info found"


df["Formatted_Education"] = df["education_list"].apply(format_education)

# Save original data + new column
df.to_csv("education_with_formatted_text.csv", index=False)
print("Saved: education_with_formatted_text.csv")
