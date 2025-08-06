import os
import json
import re
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Set style for better looking tables
plt.style.use('default')

def create_table_image(df, title, filename, figsize=(12, 8), dedup_cols=None):
    """Create a styled table image from a DataFrame"""

    # Hide duplicate entries for first col
    if dedup_cols:
        # Don't change the original
        df = df.copy()
        for col_i in dedup_cols:
            col_name = df.columns[col_i]
            empty_condition = df[col_name].shift() == df[col_name]

            if col_i > 0:
                prev_col_i = col_i - 1
                prev_col_name = df.columns[prev_col_i]
                prev_empty = (df[prev_col_name] == '')
                empty_condition &= prev_empty

            df.loc[empty_condition, col_name] = ''

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Automatically set column widths based on content
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f3f3f3')

    # plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def extract_from_file(filepath):
    with open(filepath, "r") as f:
        try:
            session = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON from {filepath}: {e}")
            return None

    trace = session.get("trace", [])
    tool_calls_detailed = []

    # Look for tool invocations
    for t in trace:
        if t.get("role") == "assistant":
            action = re.search(r"```(?:\w*\n)?(.*?)```\Z", t["content"], re.DOTALL)
            if action:
                action = action.group(1).strip()
                # Try to match function and arguments: function_name("arg1", 123)
                match = re.match(r"(\w+)\((.*)\)", action, re.DOTALL)
                if match:
                    tool_name = match.group(1)
                    args = match.group(2).strip()
                    tool_calls_detailed.append({
                        "tool": tool_name,
                        "args": args
                    })


    invalid_action = []
    # Look for cases where agent fails to ouput a valid action:
    for t in trace:
        if t.get("role") == "env":
            m = t["content"].strip()
            if m.startswith("Invalid action:"):
                invalid_action.append(m)

    # Extract trial type as the parent of the parent directory
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
    # Extract agent_name_model_name as the parent directory
    agent_llm_name = os.path.basename(os.path.dirname(filepath))

    # Extract round index from filename (e.g., run_0.json -> 0)
    filename = os.path.basename(filepath)
    round_index = None
    match = re.match(r"run_(\d+)\.json", filename)
    if match:
        round_index = int(match.group(1))

    def get_accuracy_field(d):
        for key in ["Localization Accuracy", "Detection Accuracy"]:
            if key in d:
                return d[key]

        result_success = d.get("success", None)
        if result_success is not None:
            return "100" if result_success else "0"

        return "Can't get it"
        # raise ValueError("No accuracy found. We only support Localization and Detection problems for now")

    return {
        "session_id": session.get("session_id"),
        "agent": session.get("agent"),
        "problem_id": session.get("problem_id"),
        "faulty_service": session.get("faulty_service"),
        "agent_llm_name": agent_llm_name,
        "exp_name": exp_name,
        "round_index": round_index,
        "start_time": datetime.fromtimestamp(session["start_time"]).isoformat(),
        "end_time": datetime.fromtimestamp(session["end_time"]).isoformat(),
        "time_taken_sec": round(session["end_time"] - session["start_time"], 2),
        "tokens_in": session["results"].get("in_tokens"),
        "tokens_out": session["results"].get("out_tokens"),
        "steps": session["results"].get("steps"),
        # TODO: FIX for other accuracy types
        "accuracy": get_accuracy_field(session["results"]),
        "tool_calls_ordered": "".join(f"\n{str(item)}" for item in tool_calls_detailed), #tool_calls_detailed,
        "num_tool_calls": len(tool_calls_detailed),
        "num_no_action": session["results"].get("steps") - len(tool_calls_detailed),
        "num_invalid_action": len(invalid_action)
    }

def build_insights_table(results_dir):
    all_records = []
    pid_dir = os.path.join(results_dir, 'pid')
    for root, dirs, files in os.walk(pid_dir):
        for filename in files:
            if not filename.endswith('.json'):
                continue
            full_path = os.path.join(root, filename)
            record = extract_from_file(full_path)
            if record:
                all_records.append(record)
    return pd.DataFrame(all_records)

def main():
    parser = argparse.ArgumentParser(description='Extract results from experiment JSON files')
    parser.add_argument('--folder', '-f',
                       default='experiment_results',
                       help='Folder containing experiment results (default: experiment_results)')
    parser.add_argument('--output', '-o',
                       help='Output CSV filename (default: {folder}/extracted_results.csv)')

    args = parser.parse_args()

    # Set default output path if not specified
    if args.output is None:
        args.output = os.path.join(args.folder, "extracted_results.csv")

    # Check if the folder exists
    if not os.path.exists(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist.")
        return

    print(f"Results will be extracted from '{args.folder}' and saved to '{args.output}'")
    # Build insights table
    df = build_insights_table(args.folder)
    if df.empty:
        print(f"No valid JSON files found in '{args.folder}'")
        return

    # Ensure results directory exists
    results_dir = os.path.join(os.path.dirname(args.output), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Sort and save results
    df = df.sort_values(by=["agent", "start_time"])
    main_csv_path = os.path.join(results_dir, os.path.basename(args.output))
    df.to_csv(main_csv_path, index=False)
    print(f"Total records processed: {len(df)}")
    print(f"Main results saved to {main_csv_path}")

    # --- Trial type comparison tables ---

    # Group by problem_id and exp_name, then pivot to show trial types as columns
    # summary_grouped = (
    #     df.groupby(['problem_id', 'exp_name', 'agent_llm_name', 'round_index'])
    #       .agg(
    #           accuracy=('accuracy', 'sum'),
    #           tokens_in=('tokens_in', 'sum'),
    #           tokens_out=('tokens_out', 'sum'),
    #           steps=('steps', 'sum'),
    #           num_tool_calls=('num_tool_calls', 'sum'),
    #           num_no_action=('num_no_action', 'sum')
    #       )
    #       .reset_index()
    # )

    table_columns = ['problem_id', 'agent_llm_name', 'exp_name', 'round_index', 'accuracy',
                     'tokens_in', 'tokens_out', 'steps', 'num_tool_calls', 'num_no_action', 'num_invalid_action']
    exp_df = df.sort_values(['problem_id', 'agent_llm_name', 'exp_name', 'round_index'])[table_columns]

    # Round numeric columns for better display
    exp_df['accuracy'] = exp_df['accuracy'].round(3)

    filename = os.path.join(results_dir, 'exp_comparison.png')
    create_table_image(exp_df, 'Comparing all experiments', filename, dedup_cols=[0, 1, 2])
    print(f"Created comparison table image in: {filename}")

    # Create one table per pid so it's easier to read
    pid_dir = os.path.join(results_dir, 'per_pid')
    os.makedirs(pid_dir, exist_ok=True)
    pids = exp_df['problem_id'].unique().tolist()
    for pid in pids:
        filtered_df = exp_df[exp_df['problem_id'] == pid]
        filename = os.path.join(pid_dir, f'{pid}.png')
        create_table_image(filtered_df, f'Comparing PID solutions for {pid}', filename, dedup_cols=[0, 1, 2])
        print(f"Created comparison table image for PID in: {filename}")

if __name__ == "__main__":
    main()