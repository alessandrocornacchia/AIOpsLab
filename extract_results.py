import os
import json
import re
import pandas as pd
from datetime import datetime

RESULTS_DIR = "./aiopslab/data/results/"
CUTOFF_TIME = 1745336296.0254824
#1745300000.2251244
# 1744825271.3439772
#1744208585.4308631
def extract_from_file(filepath):
    with open(filepath, "r") as f:
        try:
            session = json.load(f)
        except json.JSONDecodeError:
            return None  # skip corrupted files

    if session.get("start_time", 0) <= CUTOFF_TIME:
        return None

    trace = session.get("trace", [])
    tool_calls_detailed = []

    for t in trace:
        if t.get("role") == "assistant":
            actions = re.findall(r"```(?:\w*\n)?(.*?)```", t["content"], re.DOTALL)
            for action in actions:
                stripped = action.strip()
                # Try to match function and arguments: function_name("arg1", 123)
                match = re.match(r"(\w+)\((.*)\)", stripped, re.DOTALL)
                if match:
                    tool_name = match.group(1)
                    args = match.group(2).strip()
                    tool_calls_detailed.append({
                        "tool": tool_name,
                        "args": args
                    })

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
        "tool_calls_ordered": tool_calls_detailed,
        "num_tool_calls": len(tool_calls_detailed)
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