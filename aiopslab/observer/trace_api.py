# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import socket
import time
import select
import subprocess
import threading
from datetime import datetime, timedelta
from collections import defaultdict

import requests
import pandas as pd

from aiopslab.observer import root_path


class TraceAPI:
    def __init__(self, namespace: str):
        self.port_forward_process = None
        self.namespace = namespace
        self.stop_event = threading.Event()

        # NOTE: it may not be jaeger-out for other apps
        node_port = self.get_nodeport("jaeger", namespace)
        if node_port:
            self.base_url = f"http://localhost:{node_port}"
        else:
            self.base_url = "http://localhost:16686"
            self.start_port_forward()

    def get_nodeport(self, service_name, namespace):
        """Fetch the NodePort for the given service."""
        try:
            result = subprocess.check_output(
                [
                    "kubectl",
                    "get",
                    "service",
                    service_name,
                    "-n",
                    namespace,
                    "-o",
                    "jsonpath={.spec.ports[0].nodePort}",
                ],
                text=True,
            )
            nodeport = result.strip()
            print(f"NodePort for service {service_name}: {nodeport}")
            return nodeport
        except subprocess.CalledProcessError as e:
            print(f"Error getting NodePort: {e.output}")
            return None

    def print_output(self, stream):
        """Thread function to print output from a subprocess stream non-blockingly."""
        while not self.stop_event.is_set():
            # Check if there is content to read
            ready, _, _ = select.select([stream], [], [], 0.1)  # 0.1-second timeout
            if ready:
                try:
                    line = stream.readline()
                    if line:
                        print(line, end="")
                    else:
                        break  # Exit if no more data and process ended
                except ValueError as e:
                    print("Stream closed:", e)
                    break
            if self.port_forward_process.poll() is not None:
                break

    def is_port_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    def start_port_forward(self):
        """Starts kubectl port-forward command to access Jaeger service."""
        for attempt in range(3):
            if self.is_port_in_use(16686):
                print(
                    f"Port 16686 is already in use. Attempt {attempt + 1} of {3}. Retrying in {3} seconds..."
                )
                time.sleep(3)
                continue

            # command = "kubectl port-forward svc/jaeger 16686:16686 -n hotel-reservation"
            command = f"kubectl port-forward svc/jaeger 16686:16686 -n {self.namespace}"
            self.port_forward_process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            thread_out = threading.Thread(
                target=self.print_output, args=(self.port_forward_process.stdout,)
            )
            thread_err = threading.Thread(
                target=self.print_output, args=(self.port_forward_process.stderr,)
            )
            thread_out.start()
            thread_err.start()
            time.sleep(3)  # Wait a bit for the port-forward to establish

            if (
                self.port_forward_process.poll() is None
            ):  # Check if the process is still running
                print("Port forwarding established successfully.")
                break
            else:
                print("Port forwarding failed. Retrying...")
        else:
            print("Failed to establish port forwarding after multiple attempts.")
        # TODO: modify this command for other microservices
        # command = "kubectl port-forward svc/jaeger 16686:16686 -n hotel-reservation"
        # self.port_forward_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # thread_out = threading.Thread(target=self.print_output, args=(self.port_forward_process.stdout,))
        # thread_err = threading.Thread(target=self.print_output, args=(self.port_forward_process.stderr,))
        # thread_out.start()
        # thread_err.start()
        # # self.output_threads.extend([thread_out, thread_err])
        # time.sleep(3)  # Wait a bit for the port-forward to establish

    def stop_port_forward(self):
        """Stops the kubectl port-forward command."""
        if self.port_forward_process:
            self.port_forward_process.terminate()  # Send SIGTERM
            self.port_forward_process.wait()  # Wait for the process to terminate
            self.stop_event.set()
            print("Set the stop event.")
            self.port_forward_process.stdout.close()
            self.port_forward_process.stderr.close()
            print("Port forwarding stopped.")

    def cleanup(self):
        """Clean up resources."""
        self.stop_port_forward()
        for thread in threading.enumerate():
            if thread != threading.current_thread():
                thread.join(timeout=5)
                if thread.is_alive():
                    print(
                        f"Thread {thread.name} could not be joined and may need to be stopped forcefully."
                    )
        print("Cleanup completed.")

    def get_services(self) -> list:
        """Fetch a list of services from the tracing API."""
        url = f"{self.base_url}/api/services"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # print(f"data: {response}")
            return data.get("data", [])
        else:
            print(f"Failed to get services: {response.status_code}")
            return []

    def get_traces(
        self,
        service_name: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = None,
    ) -> list:
        """
        Fetch traces for a specific service between start_time and end_time.
        If limit is not specified, all available traces are fetched.
        """
        # Calculate the lookback in milliseconds.
        lookback = int((datetime.now() - start_time).total_seconds())

        url = f"{self.base_url}/api/traces?service={service_name}&lookback={lookback}s"
        if limit is not None:
            url += f"&limit={limit}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.RequestException as e:
            print(f"Failed to get traces for {service_name}: {e}")
            return []

    def extract_traces(
        self, start_time: datetime, end_time: datetime, limit: int = None
    ) -> list:
        """
        Extract traces for all services between start_time and end_time.
        """
        services = self.get_services()
        print(f"services: {services}")
        all_traces = []
        # Check if services is None - sometimes Jaeger's sampling
        # will lead the number of traces very small or even none
        if services is None:
            print("No services found.")
            return all_traces
        for service in services:
            if service == "jaeger-all-in-one":  # Skip utility service
                continue
            traces = self.get_traces(
                service_name=service,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )
            for trace in traces:
                for span in trace["spans"]:
                    span[
                        "serviceName"
                    ] = service  # Directly associate service name with each span
                all_traces.append(trace)  # Collect the trace with service name included
        self.cleanup()
        print("Cleanup completed.")
        # print(f"all_traces: {all_traces}")
        return all_traces

    # def process_traces(self, traces) -> pd.DataFrame:
    #     """Process raw traces data into a structured DataFrame."""
    #     trace_id_list = []
    #     service_name_list = []
    #     operation_name_list = []
    #     start_time_list = []
    #     duration_list = []
    #     parent_span_list = []

    #     for trace in traces:
    #         trace_id = trace["traceID"]
    #         for span in trace["spans"]:
    #             trace_id_list.append(trace_id)
    #             service_name_list.append(
    #                 span["serviceName"]
    #             )  # Use the correct service name from the span
    #             operation_name_list.append(span["operationName"])
    #             start_time_list.append(span["startTime"])
    #             duration_list.append(span["duration"])
    #             parent_span = "ROOT"
    #             if "references" in span:
    #                 for ref in span["references"]:
    #                     if ref["refType"] == "CHILD_OF":
    #                         parent_span = ref["spanID"]
    #                         break
    #             parent_span_list.append(parent_span)

    #     df = pd.DataFrame(
    #         {
    #             "trace_id": trace_id_list,
    #             "service_name": service_name_list,
    #             "operation_name": operation_name_list,
    #             "start_time": start_time_list,
    #             "duration": duration_list,
    #             "parent_span": parent_span_list,
    #         }
    #     )
    #     return df
    def process_traces(self, traces):
        """
        Convert a list of Jaeger traces into a pandas DataFrame with trace analytics.

        This replaces the old `process_traces()` method from the original code.
        It calculates timing information, error states, and service relationships.
        
        Returns a DataFrame with columns:
            - 'trace_id'
            - 'latency'
            - 'total_spans'
            - 'contains_errors'
            - 'error_rate'
            - 'error_messages'
            - 'operation'
            - 'services'
            - 'longest_span_service'
            - 'longest_span_duration'
            - 'mean_span_duration'
        """
        data = []

        for trace in traces:
            traceid = trace['traceID']
            spans = trace['spans']
            processes = trace.get('processes', {})

            # Build parent-to-children map
            span_id_map = {span['spanID']: span for span in spans}
            parent_to_children = {}
            for sp in spans:
                for ref in sp.get('references', []):
                    if ref['refType'] == 'CHILD_OF':
                        parent_id = ref['spanID']
                        parent_to_children.setdefault(parent_id, []).append(sp['spanID'])

            # Merge intervals helper
            def merge_intervals(intervals):
                if not intervals:
                    return []
                intervals.sort(key=lambda x: x[0])
                merged = [intervals[0]]
                for current in intervals[1:]:
                    last = merged[-1]
                    if current[0] <= last[1]:
                        merged[-1] = (last[0], max(last[1], current[1]))
                    else:
                        merged.append(current)
                return merged

            # Recursively collect child intervals
            def get_descendant_intervals(span_id, parent_start, parent_end):
                intervals = []
                child_ids = parent_to_children.get(span_id, [])
                for child_id in child_ids:
                    child_span = span_id_map[child_id]
                    child_start = child_span['startTime']
                    child_end = child_start + child_span['duration']

                    # Clip the child's interval to the parent's interval
                    clipped_start = max(child_start, parent_start)
                    clipped_end = min(child_end, parent_end)
                    if clipped_start < clipped_end:
                        intervals.append((clipped_start, clipped_end))
                        intervals.extend(
                            get_descendant_intervals(child_id, clipped_start, clipped_end)
                        )
                return merge_intervals(intervals)

            # Calculate total latency
            start_times = [s['startTime'] for s in spans]
            durations = [s['duration'] for s in spans]
            end_times = [start + dur for start, dur in zip(start_times, durations)]
            latency = max(end_times) - min(start_times)

            # Basic stats
            spans_num = len(spans)
            error_bool = False
            error_count = 0
            error_msgs = defaultdict(int)
            service_names = set()

            # Root span detection
            root_span = next((s for s in spans if not s.get('references')), None)
            if not root_span:
                # fallback: earliest start
                root_span = min(spans, key=lambda s: s['startTime'])
            operation = root_span['operationName']

            total_self_duration = 0
            longest_self_time = 0
            longest_self_time_span = None

            # Process each span
            for sp in spans:
                process_id = sp['processID']
                # processes might not exist or have a processID
                service_name = processes.get(process_id, {}).get('serviceName', 'unknown-service')
                service_names.add(service_name)

                # Check for errors in logs
                for log in sp.get('logs', []):
                    for field in log.get('fields', []):
                        if field.get('key') == 'event' and field.get('value') == 'exception':
                            error_count += 1
                            error_bool = True
                            # pull an error message if available
                            for err_field in log.get('fields', []):
                                if err_field.get('key') == 'exception.message':
                                    error_msgs[err_field.get('value')] += 1
                            break

                # Self duration
                span_start = sp['startTime']
                span_end = span_start + sp['duration']
                descendant_intervals = get_descendant_intervals(sp['spanID'], span_start, span_end)
                descendant_time = sum(e - st for st, e in descendant_intervals)
                self_time = sp['duration'] - descendant_time
                if self_time < 0:
                    self_time = 0

                total_self_duration += self_time

                if self_time > longest_self_time:
                    longest_self_time = self_time
                    longest_self_time_span = sp

            error_rate = error_count / spans_num if spans_num else 0
            error_messages_str = '; '.join(error_msgs)
            services_str = '; '.join(service_names)
            longest_span_service = None
            if longest_self_time_span:
                proc_id = longest_self_time_span['processID']
                longest_span_service = processes.get(proc_id, {}).get('serviceName', 'unknown-service')

            mean_span_duration = total_self_duration / spans_num if spans_num else 0

            data.append({
                'trace_id': traceid,
                'latency': latency,
                'total_spans': spans_num,
                'contains_errors': error_bool,
                'error_rate': error_rate,
                'error_messages': error_messages_str,
                'operation': operation,
                'services': services_str,
                'longest_span_service': longest_span_service,
                'longest_span_duration': longest_self_time,
                'mean_span_duration': mean_span_duration,
            })

        return pd.DataFrame(data)

    def analyze_trace(self, all_traces, trace_id) -> str:
        """
        Analyzes a single Jaeger trace (by trace_id) from a collection of traces.
        Returns a string representation of the key metrics.
        """
        # Find the trace
        trace = None
        for t in all_traces:
            if t['traceID'] == trace_id:
                trace = t
                break

        if trace is None:
            raise KeyError(f"Trace {trace_id} not found in the provided collection.")

        spans = trace['spans']
        processes = trace.get('processes', {})

        # Total duration
        start_time = min(s['startTime'] for s in spans)
        end_time = max(s['startTime'] + s['duration'] for s in spans)
        total_duration = end_time - start_time

        # Number of spans
        total_spans = len(spans)

        # Error info
        error_count = 0
        error_messages = defaultdict(int)
        for sp in spans:
            for lg in sp.get('logs', []):
                for field in lg.get('fields', []):
                    if field.get('key') == 'event' and field.get('value') == 'exception':
                        error_count += 1
                        for ef in lg.get('fields', []):
                            if ef.get('key') == 'exception.message':
                                error_messages[ef.get('value')] += 1
                        break
        error_rate = error_count / total_spans if total_spans else 0

        # Request flow
        spans_dict = {s['spanID']: s for s in spans}
        parent_to_children = defaultdict(list)
        for sp in spans:
            for ref in sp.get('references', []):
                if ref['refType'] == 'CHILD_OF':
                    parent_to_children[ref['spanID']].append(sp)

        # compute self duration
        def compute_self_duration(span):
            child_spans = parent_to_children.get(span['spanID'], [])
            child_intervals = []
            for c in child_spans:
                c_start = c['startTime']
                c_end = c_start + c['duration']
                child_intervals.append((c_start, c_end))

            # merge overlapping intervals
            merged = []
            for interval in sorted(child_intervals):
                if not merged or merged[-1][1] <= interval[0]:
                    merged.append(list(interval))
                else:
                    merged[-1][1] = max(merged[-1][1], interval[1])

            total_child_time = sum(e - st for st, e in merged)
            return span['duration'] - total_child_time

        # build request flow
        request_flow = []
        def get_service_name(process_id):
            return processes.get(process_id, {}).get('serviceName', 'unknown-service').split(':')[-1]

        def process_span(sp):
            service_name = get_service_name(sp['processID'])
            operation = sp['operationName']
            st = sp['startTime']
            dur = sp['duration']

            parent_span_id = None
            for ref in sp.get('references', []):
                if ref['refType'] == 'CHILD_OF':
                    parent_span_id = ref['spanID']
                    break
            parent_service = None
            if parent_span_id and parent_span_id in spans_dict:
                parent_service = get_service_name(spans_dict[parent_span_id]['processID'])

            flow_item = {
                'from': parent_service,
                'to': service_name,
                'operation': operation,
                'start_time': st,
                'duration': dur
            }
            # only record if we have an actual cross-service or parent->child transition
            request_flow.append(flow_item)

        # fill in self_duration and flow
        for sp in spans:
            sp['self_duration'] = compute_self_duration(sp)
            process_span(sp)

        # find the span with the longest self_duration
        longest_span = max(spans, key=lambda x: x['self_duration'])
        longest_span_info = {
            'spanID': longest_span['spanID'],
            'operationName': longest_span['operationName'],
            'serviceName': get_service_name(longest_span['processID']),
            'self_duration': longest_span['self_duration']
        }

        # sort request_flow by start_time
        request_flow.sort(key=lambda x: x['start_time'])

        summary = {
            'total_duration in microseconds': total_duration,
            'total_spans': total_spans,
            'error_rate': error_rate,
            'error_messages': dict(error_messages),
            'request_flow': request_flow,
            'longest_span': longest_span_info
        }
        return str(summary)

    def save_traces(self, df, path) -> str:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"traces_{int(time.time())}.csv")
        df.to_csv(file_path, index=False)
        self.cleanup() # Stop port-forwarding after traces are exported
        return f"Traces data exported to: {file_path}"


if __name__ == "__main__":
    tracer = TraceAPI(namespace="test-hotel-reservation")
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=9)  # Example time window
    traces = tracer.extract_traces(start_time, end_time)
    df_traces = tracer.process_traces(traces)
    save_path = root_path / "trace_output"
    tracer.save_traces(df_traces, save_path)