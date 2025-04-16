import os
import json
import re
import pandas as pd
from datetime import datetime

RESULTS_DIR = "./aiopslab/data/results"
CUTOFF_TIME = 1744208585.4308631
# 1744812465.4460268

def extract_from_file(filepath):
    with open(filepath, "r") as f:
        try:
            session = json.load(f)
        except json.JSONDecodeError:
            return None  # skip corrupted files

    if session.get("start_time", 0) <= CUTOFF_TIME:
        return None

    trace = session.get("trace", [])
    tools_used = []
    tool_calls = 0

    for t in trace:
        if t.get("role") == "assistant":
            actions = re.findall(r"```(?:\w*\n)?(.*?)```", t["content"], re.DOTALL)
            for action in actions:
                cleaned = action.strip().split('(')[0].strip()
                tools_used.append(cleaned)
                tool_calls += 1

    return {
        "session_id": session.get("session_id"),
        "agent": session.get("agent"),
        "problem_id": session.get("problem_id"),
        "start_time": datetime.fromtimestamp(session["start_time"]).isoformat(),
        "end_time": datetime.fromtimestamp(session["end_time"]).isoformat(),
        "time_taken_sec": round(session["end_time"] - session["start_time"], 2),
        "tokens_in": session["results"].get("in_tokens"),
        "tokens_out": session["results"].get("out_tokens"),
        "steps": session["results"].get("steps"),
        "localization_accuracy": session["results"].get("Localization Accuracy"),
        "tools_used_order": tools_used,
        "num_tool_calls": tool_calls
    }

def build_insights_table(results_dir):
    all_records = []

    for filename in os.listdir(results_dir):
        if not filename.endswith(".json"):
            continue
        full_path = os.path.join(results_dir, filename)
        record = extract_from_file(full_path)
        if record:
            all_records.append(record)

    return pd.DataFrame(all_records)

# Example usage
df = build_insights_table(RESULTS_DIR)
df= df.sort_values(by=["agent", "start_time"])
df.to_csv("extracted_results.csv", index=False)