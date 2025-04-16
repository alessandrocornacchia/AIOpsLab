# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base class for task actions."""

import os
import pandas as pd
import numpy as np
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
import json
from datetime import datetime, timedelta
from aiopslab.utils.actions import action, read, write
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.shell import Shell

# from aiopslab.observer import initialize_pod_and_service_lists
from aiopslab.observer.metric_api import PrometheusAPI
from aiopslab.observer.trace_api import TraceAPI
import openai


class TaskActions:
    """Base class for task actions."""

    @staticmethod
    @read
    def get_logs(namespace: str, service: str) -> str:
        """
        Collects relevant log data from a pod using Kubectl. Use the service name without the pod suffix.

        Args:
            namespace (str): The namespace in which the service is running.
            service (str): The name of the service.

        Returns:
            str | dict | list[dicts]: Log data as a structured object or a string.
        """
        kubectl = KubeCtl()
        try:
            if namespace == "test-social-network":
                user_service_pod = kubectl.get_pod_name(namespace, f"app={service}")
            elif namespace == "test-hotel-reservation":
                user_service_pod = kubectl.get_pod_name(
                    namespace, f"io.kompose.service={service}"
                )
            else:
                raise Exception
            logs = kubectl.get_pod_logs(user_service_pod, namespace)
        except Exception as e:
            return "Error: Your service/namespace does not exist. Use kubectl to check."

        logs = "\n".join(logs.split("\n"))

        return logs

    @staticmethod
    @action
    def exec_shell(command: str) -> str:
        """
        Execute any shell command in a predefined debugging environment.
        Note: this is NOT A STATEFUL OR INTERACTIVE shell session. So you cannot
        execute commands like "kubectl edit".

        Args:
            command (str): The command to execute.

        Returns:
            str: The output of the command.
        """
        if "kubectl edit" in command or "edit svc" in command:
            return "Error: Cannot use `kubectl edit`. Use `kubectl patch` instead."

        return Shell.exec(command)

    @staticmethod
    @read
    def get_metrics(namespace: str, duration: int = 5) -> str:
        """
        Collects metrics data from the service using Prometheus.

        Args:
            namespace (str): The namespace in which the service is running.
            duration (int): The number of minutes from now to start collecting metrics until now.

        Returns:
            str: Path to the directory where metrics are saved.
        """
        prometheus_url = (
            "http://localhost:32000"  # Replace with your Prometheus server URL
        )
        prometheus_api = PrometheusAPI(prometheus_url, namespace)
        prometheus_api.initialize_pod_and_service_lists(namespace)

        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=duration)
        save_path = os.path.join(os.getcwd(), "metrics_output")

        # Export all metrics and save to the specified path
        save_dir_str = prometheus_api.export_all_metrics(
            start_time=start_time, end_time=end_time, save_path=save_path, step=1
        )

        return save_dir_str
    
    @staticmethod
    @read
    def read_metrics(file_path: str) -> str:
        """
        Reads and returns metrics from a specified CSV file, adding a time-series column.

        Args:
            file_path (str): Path to the metrics file (CSV format).

        Returns:
            str: The requested metrics with the time-series column or an error message.
        """
        if not os.path.exists(file_path):
            return {"error": f"Metrics file '{file_path}' not found."}

        try:
            df_metrics = pd.read_csv(file_path)

            # Ensure the 'timestamp' column exists
            if 'timestamp' in df_metrics.columns:
                df_metrics['timestamp'] = pd.to_datetime(df_metrics['timestamp'], unit='s')  # Convert to datetime
                df_metrics['time_series'] = df_metrics['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')  # Create time series

            return df_metrics.to_string(index=False)

        except Exception as e:
            return f"Failed to read metrics: {str(e)}"

    @staticmethod
    # @read
    def get_metric_summary(file_path: str) -> str:
        """
        Please call get_metrics() before calling this function to generate the CSV files.
        Provides a statistical summary of the metric in the file, including max/min metric services.

        Args:
            file_path (str): Path to the metrics file.

        Returns:
            str: A formatted string containing the summary statistics.
        """
        if not os.path.exists(file_path):
            return {"error": f"Metrics file '{file_path}' not found."}

        try:
            df = pd.read_csv(file_path)

            # Ensure the 'timestamp' column exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # Convert to datetime
                df['time_series'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')  # Create time series

        except Exception as e:
            return f"Failed to read metrics: {str(e)}"

        if 'cmdb_id' not in df.columns or 'value' not in df.columns:
            return "Error: Metrics file does not contain required columns ('cmdb_id' and 'value')."

        # Compute statistical summary
        summary = df['value'].describe(percentiles=[0.25, 0.5, 0.75])

        # Identify CMDB IDs with max and min values
        max_value = df['value'].max()
        min_value = df['value'].min()
        max_cmdb = df[df['value'] == max_value]['cmdb_id'].values[0]
        min_cmdb = df[df['value'] == min_value]['cmdb_id'].values[0]

        # Identify top 5 CMDBs by average value
        top_cmdbs = df.groupby('cmdb_id')['value'].mean().nlargest(5)

        # Format output as a string
        summary_str = (
            f"Statistical Summary for Metric in {file_path}:\n"
            f"-----------------------------------------\n"
            f"Count: {int(summary['count'])}\n"
            f"Mean: {summary['mean']:.5f}\n"
            f"Std Dev: {summary['std']:.5f}\n"
            f"Min: {summary['min']:.5f} (Service: {min_cmdb})\n"
            f"25th Percentile: {summary['25%']:.5f}\n"
            f"Median (50%): {summary['50%']:.5f}\n"
            f"75th Percentile: {summary['75%']:.5f}\n"
            f"Max: {summary['max']:.5f} (Service: {max_cmdb})\n"
            f"\nTop 5 Services by Average Value:\n"
            + "\n".join([f"- {cmdb}: {val:.5f}" for cmdb, val in top_cmdbs.items()])
        )

        return summary_str

    @staticmethod
    # @read
    def detect_high_cpu_pods(csv_file, threshold_cpu=0.5):
        """
        Please call get_metrics() before calling this function to generate the CSV files.
        Detect all instances where CPU usage exceeded a specified threshold.

        Args:
            csv_file (str): Path to the CSV file containing CPU metrics
            threshold_cpu (float): CPU usage threshold in cores (default: 0.5 cores)

        Returns:
            list: List of tuples containing (timestamp, pod_name, cpu_usage)
        """
        import pandas as pd

        df = pd.read_csv(csv_file)

        # Filter for CPU usage metrics
        cpu_df = df[df['kpi_name'] == 'container_cpu_usage_seconds_total'].copy()

        # Extract pod name from cmdb_id
        cpu_df['pod_name'] = cpu_df['cmdb_id'].apply(lambda x: x.split('.', 1)[1])

        # Convert timestamp to readable datetime if needed
        if 'timestamp' in cpu_df.columns:
            cpu_df['timestamp'] = pd.to_datetime(cpu_df['timestamp'], unit='s')

        # Filter where CPU value exceeds the threshold
        high_cpu_df = cpu_df[cpu_df['value'] > threshold_cpu]

        # Sort by value (optional)
        high_cpu_df = high_cpu_df.sort_values(by='value', ascending=False)

        if not high_cpu_df.empty:
            result = "High CPU usage instances:\n"
            for _, row in high_cpu_df.iterrows():
                result += f"Time: {row['timestamp']}, Pod: {row['pod_name']}, CPU: {row['value']:.3f} cores\n"
            return result
        else:
            return "No high CPU usage instances found."
    
    @staticmethod
    # @read
    def analyze_metric(file_path: str) -> str:
        """
        Please call get_metrics() before calling this function to generate the CSV files.
        Analyzes the given metric and returns a summary.
        
        Args:
            file_path (str): Path to the metrics CSV file.

        Returns:
            str: A text analysis of the metric data.
        """
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found."

        try:
            df = pd.read_csv(file_path)

            # Convert timestamp if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df['time_series'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')


            # Downsample for GPT input if too large
            sample_df = df.sample(n=min(len(df), 1000), random_state=42)
            
            # Select relevant columns
            context_df = sample_df[['timestamp', 'cmdb_id', 'value', 'kpi_name']] if 'kpi_name' in df.columns else sample_df[['timestamp', 'cmdb_id', 'value']]
            context_csv = context_df.to_csv(index=False)

            # Compose prompt
            prompt = f"""
                        You are a metrics analyst assistant. Analyze the following metrics data and provide any key patterns and outliers. 
                        Suggest exactly what service(s) need attention or further investigation, that's it.

                        Metric file: {file_path}
                        Metric name: {df['kpi_name'][0] or "Not specified"}

                        Here are the sampled records:

                        {context_csv}
                    """

            # Call OpenAI
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant skilled at analyzing metrics from time-series data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Failed to analyze metric: {str(e)}"

    @staticmethod
    # @read
    def cluster_metrics(file_path: str, n_clusters: int = 3) -> str:
        """
        Clusters time series metrics using k-shape clustering.

        Args:
            file_path (str): Path to the CSV file containing time-series metrics.
            n_clusters (int): Number of clusters to create.

        Returns:
            str: Cluster mapping as a string or an error message.
        """
        
        if not os.path.exists(file_path):
            return "Error: Metrics file not found."

        try:
            # Load CSV
            df = pd.read_csv(file_path)

            # Ensure necessary columns exist
            required_columns = {'timestamp', 'cmdb_id', 'value'}
            if not required_columns.issubset(df.columns):
                return f"Error: CSV file must contain {required_columns} columns."

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

            # Ensure 'value' column is numeric
            df['value'] = pd.to_numeric(df['value'], errors='coerce')  # Convert non-numeric to NaN
            df.dropna(subset=['value'], inplace=True)  # Remove rows where value is NaN
            
            if df.empty:
                return "Error: No valid numerical time-series data available."
            
            # Pivot the data so each `cmdb_id` becomes a column (one time series per entity)
            df_pivot = df.pivot(index='timestamp', columns='cmdb_id', values='value')

            if df_pivot.empty:
                return "Error: No valid time series extracted."

            # Resample to a uniform frequency (fill missing timestamps)
            df_pivot = df_pivot.resample('min').mean().interpolate()  # Resample to 1-minute intervals, interpolate missing values
            df_pivot = df_pivot.fillna(0)

            # Extract time series values
            metric_names = df_pivot.columns.tolist()  # Unique cmdb_id values
            time_series = [df_pivot[col].values for col in metric_names]  # Extract values
            
            if not time_series:
                return "Error: No valid time series found after cleaning."

            # Convert list of series to uniform shape
            X = to_time_series_dataset(time_series)  # Auto-pads/truncates to a uniform shape
            
            # Scale the time series using RobustScaler (prevents outliers from being ignored)
            scaler = TimeSeriesScalerMeanVariance()
            X_scaled = np.array([scaler.fit_transform(x.reshape(-1, 1)).flatten() for x in X])

            # Perform k-shape clustering
            ks = KShape()
            cluster_labels = ks.fit_predict(X_scaled)

            # Create mapping of clusters to metric names
            clusters = {}
            for label, name in zip(cluster_labels, metric_names):
                clusters.setdefault(label, []).append(name)  # Store only cmdb_id

            # Convert dictionary to a string output
            result_str = "\n".join([f"Cluster {label}: {', '.join(names)}" for label, names in clusters.items()])
            return result_str

        except Exception as e:
            return f"Error: Failed to cluster metrics - {str(e)}"

    @staticmethod
    @read
    def get_traces(namespace: str, duration: int = 5) -> str:
        """
        Collects trace data from the service using Jaeger.

        Args:
            namespace (str): The namespace in which the service is running.
            duration (int): The number of minutes from now to start collecting traces until now.

        Returns:
            str: Path to the directory where traces are saved.
        """
        # jaeger_url = "http://localhost:16686"
        print(namespace)
        trace_api = TraceAPI(namespace=namespace)

        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=duration)

        traces = trace_api.extract_traces(start_time=start_time, end_time=end_time)
        df_traces = trace_api.process_traces(traces)
        save_path = os.path.join(os.getcwd(), "trace_output")

        return trace_api.save_traces(df_traces, save_path)
        # return f"Trace data exported to: {save_path}"

    @staticmethod
    @read
    def read_traces(file_path: str) -> str:
        """
        Reads and returns traces from a specified CSV file.

        Args:
            file_path (str): Path to the traces file (CSV format).

        Returns:
            str: The requested traces or an error message.
        """
        if not os.path.exists(file_path):
            return {"error": f"Traces file '{file_path}' not found."}

        try:
            df_traces = pd.read_csv(file_path)

            return df_traces.to_string(index=False)

        except Exception as e:
            return f"Failed to read traces: {str(e)}"

    @staticmethod
    # @read
    def analyze_specific_trace(
        namespace: str, 
        trace_id: str,
        duration: int = 5, 
    ) -> str:
        """
        Please call get_traces() before calling this function to generate the CSV file.
        Analyzes a Jaeger trace from the given namespace. It extracts traces from Jaeger
        within the specified duration (in minutes), then calls the analyze_trace
        method on a particular trace_id.

        Args:
            namespace (str): The Kubernetes namespace of Jaeger and your services.
            trace_id (str): The specific trace to analyze.
            duration (int): Time window in minutes from now going backward to collect traces.
            
        Returns:
            str: The analysis for the chosen trace.
        """
        print(f"Analyzing trace(s) in namespace: {namespace} for last {duration} minutes.")
        trace_api = TraceAPI(namespace=namespace)

        # Compute time window
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=duration)

        # Extract all traces from Jaeger within the time window
        traces = trace_api.extract_traces(start_time=start_time, end_time=end_time)
        if not traces:
            return "No traces found in the given time window."

        # If user didn't specify trace_id, pick the first trace by default
        if not trace_id:
            trace_id = traces[0]['traceID']
            print(f"No trace_id specified. Using the first available trace: {trace_id}")

        # Perform the analysis
        try:
            analysis_result = trace_api.analyze_trace(traces, trace_id)
            if isinstance(analysis_result, str):
                # If your analyze_trace returns a string, load it back to dict (if valid JSON-ish)
                try:
                    analysis_dict = eval(analysis_result)  # or json.loads(...) if valid JSON
                except Exception:
                    return analysis_result  # fallback

                # Then convert to a nice JSON string
                pretty_str = json.dumps(analysis_dict, indent=2)
                return pretty_str
            elif isinstance(analysis_result, dict):
                # If your analyze_trace can return dict directly
                return json.dumps(analysis_result, indent=2)
            else:
                return str(analysis_result)
        except KeyError as ke:
            return str(ke)
    
    @staticmethod
    # @read
    def get_traces_summary(file_path: str) -> str:
        """
        Please call get_traces() before calling this function to generate the CSV file.
        Reads a traces file (CSV) and returns a detailed statistical summary including
        average latency, total requests, error rate, and longest spans, along with trace IDs.
        
        Args:
            file_path (str): Path to the traces file.
        
        Returns:
            str: A formatted string containing trace summary statistics.
        """
        if not os.path.exists(file_path):
            return f"Error: Traces file '{file_path}' not found."
        
        try:
            # Load the traces CSV file
            df = pd.read_csv(file_path)
            
            # Compute statistical summary
            total_traces = len(df)
            avg_latency = df["latency"].mean()
            max_latency = df["latency"].max()
            min_latency = df["latency"].min()
            error_rate = df["error_rate"].mean()
            total_errors = df[df["contains_errors"] == True].shape[0]
            most_common_operation = df["operation"].mode()[0]
            most_common_service = df["services"].mode()[0]
            
            # Identify trace IDs for max values
            max_latency_trace = df.loc[df["latency"].idxmax(), "trace_id"]
            min_latency_trace = df.loc[df["latency"].idxmin(), "trace_id"]
            max_span_trace = df.loc[df["longest_span_duration"].idxmax(), "trace_id"]
            
            # Identify the longest span service
            longest_span_service = df.loc[df["longest_span_duration"].idxmax(), "longest_span_service"]
            max_span_duration = df["longest_span_duration"].max()
            mean_span_duration = df["mean_span_duration"].mean()
            
            # Format summary as a string
            summary_str = (
                f"Trace Summary Report\n"
                f"---------------------------------\n"
                f"Total Traces: {total_traces}\n"
                f"Average Latency: {avg_latency:.2f} ms\n"
                f"Max Latency: {max_latency:.2f} ms (Trace ID: {max_latency_trace})\n"
                f"Min Latency: {min_latency:.2f} ms (Trace ID: {min_latency_trace})\n"
                f"Error Rate: {error_rate:.2%}\n"
                f"Total Errors: {total_errors}\n"
                f"Most Common Operation: {most_common_operation}\n"
                f"Most Common Service: {most_common_service}\n"
                f"Longest Span Service: {longest_span_service}\n"
                f"Max Span Duration: {max_span_duration:.2f} ms (Trace ID: {max_span_trace})\n"
                f"Mean Span Duration: {mean_span_duration:.2f} ms\n"
            )
            
            return summary_str
        except Exception as e:
            return f"Error processing traces file: {str(e)}"

    @staticmethod
    # @read
    def analyze_traces(file_path: str) -> str:
        """
        Please call get_traces() before calling this function to generate the CSV file.
        Analyzes trace data and returns an insightful summary.

        Args:
            file_path (str): Path to the CSV file containing trace data.

        Returns:
            str: Textual summary of the trace data.
        """
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found."

        try:
            df = pd.read_csv(file_path)

            context_csv = df.to_csv(index=False)

            # Compose prompt
            prompt = f"""
    You are a tracing performance expert. Analyze the following trace data to detect:
    - Performance bottlenecks
    - High latency spans
    - Services or operations that might need optimization
    - Any unusual trace patterns

    Trace file: {file_path}

    Here is the trace data:

    {context_csv}
            """

            # GPT call
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant skilled in analyzing distributed trace data and identifying performance issues."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Failed to analyze trace data: {str(e)}"

    @staticmethod
    # @read
    # NOTE: disabled for now, since seems like a cheat for code changes
    def get_microservice_repo_diff(start: int, end: int, token=None) -> list[dict]:
        pass
        # """
        # Fetch the latest commits and their diffs from a GitHub repository.

        # Args:
        #     start (int): The start timestamp.
        #     end (int): The end timestamp.
        #     token: GitHub personal access token for authenticated requests (optional).

        # Returns:
        #     A list of commit messages and other details.
        # """
        # api_url = f"https://api.github.com/repos/owner/repo/commits"
        # headers = {}

        # if token:
        #     headers["Authorization"] = f"token {token}"

        # try:
        #     response = requests.get(api_url, headers=headers)
        #     response.raise_for_status()  # Raise an error for bad responses
        #     commits = response.json()

        #     for commit in commits:
        #         sha = commit["sha"]
        #         # Fetch diff details for each commit
        #         diff_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
        #         diff_response = requests.get(diff_url, headers=headers)
        #         diff_response.raise_for_status()
        #         diff_data = diff_response.json()

        #         # Extract relevant commit information and diff
        #         commit_info = {
        #             "sha": sha,
        #             "message": commit["commit"]["message"],
        #             "author": commit["commit"]["author"]["name"],
        #             "date": commit["commit"]["author"]["date"],
        #             "files": [],
        #         }

        #         # Process the diff data
        #         for file in diff_data["files"]:
        #             commit_info["files"].append(
        #                 {
        #                     "filename": file["filename"],
        #                     "status": file["status"],
        #                     "additions": file["additions"],
        #                     "deletions": file["deletions"],
        #                     "changes": file["changes"],
        #                     "patch": file["patch"],
        #                 }
        #             )

        #         commit_details.append(commit_info)

        #     return commit_details

        # except requests.RequestException as e:
        #     print(f"An error occurred: {e}")
        #     return []

if __name__ == "__main__":
    print(TaskActions.analyze_traces('/home/ubuntu/iliyas/AIOpsLab/trace_output/traces_1742771893.csv'))
    