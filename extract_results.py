import os
import json
import re
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

# Set style for better looking tables
plt.style.use('default')
plt.rcParams['savefig.dpi'] = 300

def create_table_image(df, title, filename, figsize=None, dedup_cols=None):
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
    # plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_per_type_tables(df, results_dir, table_type, folder_name = None):
    """Create a table per {table_type} from the given DataFrame."""

    folder_name = folder_name if folder_name else f'per_{table_type}'
    table_dir = os.path.join(results_dir, folder_name)

    os.makedirs(table_dir, exist_ok=True)
    types = df[table_type].unique().tolist()
    for t in types:
        filename = os.path.join(table_dir, f'{t}.png')
        filtered_df = df[df[table_type] == t]
        filtered_df = filtered_df.drop(columns=[table_type])
        create_table_image(filtered_df, f'{t}', filename, dedup_cols=[0, 1])
        print(f"Created comparison table image for {table_type} in: {filename}")

def plot_remind_actions_per_type(df, type, x_axis, folder):
    folder = os.path.join(folder, f'remind_actions_missing_per_{type}')
    os.makedirs(folder, exist_ok=True)
    for i, (name, df) in enumerate(df.groupby(type)):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df, x=x_axis, y='percent_missing_actions', hue='exp_name', ax=ax)
        ax.set_title(f"{name} - missing_actions calls per experiment")
        ax.set_xlabel(x_axis)
        ax.set_ylabel('Percentage of missing actions relative to step count')
        plt.tight_layout()
        plt.setp(plt.gca().get_xticklabels(), rotation=45, ha='right')
        filename = os.path.join(folder, f"{name}_missing_calls_per_{x_axis}.png")
        plt.savefig(filename, bbox_inches='tight')
        print(f"Created missing actions plot image in: {filename}")
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


    invalid_actions = []
    # Look for cases where agent fails to ouput a valid action:
    for t in trace:
        if t.get("role") == "env":
            m = t["content"].strip()
            if m.startswith("Invalid action:"):
                invalid_actions.append(m)

    # Extract trial type as the parent of the parent directory
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
    # Extract agent_name_model_name as the parent directory
    agent = os.path.basename(os.path.dirname(filepath))

    # Extract round index from filename (e.g., run_0.json -> 0)
    filename = os.path.basename(filepath)
    round_index = None
    match = re.match(r"run_(\d+)\.json", filename)
    if match:
        round_index = int(match.group(1))

    def get_accuracy_field(d):
        for key in ["Detection Accuracy", "Localization Accuracy"]:
            if key in d:
                return d[key]

        result_success = d.get("success", None)
        if result_success is not None:
            return "Correct" if result_success else "Incorrect"

        return "Can't get it"
        # raise ValueError("No accuracy found. We only support Localization and Detection problems for now")

    return {
        "session_id": session.get("session_id"),
        "agent": session.get("agent"),
        "pid": session.get("problem_id"),
        "faulty_service": session.get("faulty_service"),
        "agent": agent,
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
        "tool_calls": len(tool_calls_detailed),
        "missing_actions": session["results"].get("steps") - len(tool_calls_detailed),
        "invalid_actions": len(invalid_actions)
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
    exp_df = build_insights_table(args.folder)
    if exp_df.empty:
        print(f"No valid JSON files found in '{args.folder}'")
        return

    # Ensure results directory exists
    results_dir = os.path.join(os.path.dirname(args.output), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Sort and save results
    exp_df = exp_df.sort_values(by=["agent", "start_time"])
    main_csv_path = os.path.join(results_dir, os.path.basename(args.output))
    exp_df.to_csv(main_csv_path, index=False)
    print(f"Total records processed: {len(exp_df)}")
    print(f"Main results saved to {main_csv_path}")


    # Plot results
    table_columns = ['pid', 'agent', 'exp_name', 'round_index', 'accuracy',
                     'tokens_in', 'tokens_out', 'steps', 'tool_calls', 'missing_actions', 'invalid_actions']
    exp_df = exp_df.sort_values(['pid', 'agent', 'exp_name', 'round_index'])[table_columns]

    # Replace accuracies with ✗ or ✓
    exp_df['accuracy'] = exp_df['accuracy'].replace({
        100.0: '✓',
        0.0: '✗',
        'Correct': '✓',
        'Incorrect': '✗'
    })

    exp_df = exp_df[~exp_df['pid'].str.contains('memory_stress', na=False)]

    filename = os.path.join(results_dir, 'exp_comparison.png')
    create_table_image(exp_df, 'Comparing all experiments', filename, dedup_cols=[0, 1, 2])
    print(f"Created comparison table image in: {filename}")

    create_per_type_tables(exp_df, results_dir, 'pid')
    create_per_type_tables(exp_df, results_dir, 'agent')

    # # Plot remind actions
    table_columns = ['pid', 'agent', 'exp_name', 'round_index', 'steps', 'missing_actions', 'invalid_actions']
    valid_agent_exp = exp_df.groupby(['pid', 'agent'])['exp_name'].apply(lambda s: {'remind_actions', 'baseline'}.issubset(s.unique()))
    valid_agent_exp = valid_agent_exp[valid_agent_exp].index
    valid_exp_df = exp_df.set_index(['pid', 'agent']).loc[valid_agent_exp].reset_index()
    valid_exp_df = valid_exp_df.sort_values(['pid', 'agent', 'exp_name', 'round_index'])[table_columns]

    create_per_type_tables(valid_exp_df, results_dir, 'pid', folder_name = "remind_actions_per_pid")
    create_per_type_tables(valid_exp_df, results_dir, 'agent', folder_name = "remind_actions_per_agent")

    # Plot tool calls
    valid_exp_df['percent_missing_actions'] = (valid_exp_df['missing_actions'] / valid_exp_df['steps']) * 100
    plot_remind_actions_per_type(valid_exp_df, 'agent', 'pid', results_dir)
    plot_remind_actions_per_type(valid_exp_df, 'pid', 'agent', results_dir)


if __name__ == "__main__":
    main()
