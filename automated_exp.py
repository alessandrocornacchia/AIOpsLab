import subprocess
import os
import sys
import logging
import json
import time
from datetime import datetime
from pathlib import Path
import argparse
from typing import List, Dict, Any

# PIDs grouped by type, extracted from the problem registry
PIDS_BY_TYPE = {
    "detection": [
        "k8s_target_port-misconfig-detection-1",
        "k8s_target_port-misconfig-detection-2",
        "k8s_target_port-misconfig-detection-3",
        "auth_miss_mongodb-detection-1",
        "revoke_auth_mongodb-detection-1",
        "revoke_auth_mongodb-detection-2",
        "user_unregistered_mongodb-detection-1",
        "user_unregistered_mongodb-detection-2",
        "misconfig_app_hotel_res-detection-1",
        "scale_pod_zero_social_net-detection-1",
        "assign_to_non_existent_node_social_net-detection-1",
        "container_kill-detection",
        "pod_failure_hotel_res-detection-1",
        "pod_kill_hotel_res-detection-1",
        "network_loss_hotel_res-detection-1",
        "network_delay_hotel_res-detection-1",
        "memory_stress_social_net-detection-1",
        "cpu_stress_hotel_res-detection-1",
        "noop_detection_hotel_reservation-1",
        "noop_detection_social_network-1",
        "redeploy_without_PV-detection-1",
        "wrong_bin_usage-detection-1",
        "operator_overload_replicas-detection-1",
        "operator_non_existent_storage-detection-1",
        "operator_invalid_affinity_toleration-detection-1",
        "operator_security_context_fault-detection-1",
        "operator_wrong_update_strategy-detection-1",
    ],
    "localization": [
        "k8s_target_port-misconfig-localization-1",
        "k8s_target_port-misconfig-localization-2",
        "k8s_target_port-misconfig-localization-3",
        "auth_miss_mongodb-localization-1",
        "revoke_auth_mongodb-localization-1",
        "revoke_auth_mongodb-localization-2",
        "user_unregistered_mongodb-localization-1",
        "user_unregistered_mongodb-localization-2",
        "misconfig_app_hotel_res-localization-1",
        "scale_pod_zero_social_net-localization-1",
        "assign_to_non_existent_node_social_net-localization-1",
        "container_kill-localization",
        "pod_failure_hotel_res-localization-1",
        "pod_kill_hotel_res-localization-1",
        "network_loss_hotel_res-localization-1",
        "network_delay_hotel_res-localization-1",
        "memory_stress_social_net-localization-1",
        "cpu_stress_hotel_res-localization-1",
        "wrong_bin_usage-localization-1",
        "operator_overload_replicas-localization-1",
        "operator_non_existent_storage-localization-1",
        "operator_invalid_affinity_toleration-localization-1",
        "operator_security_context_fault-localization-1",
        "operator_wrong_update_strategy-localization-1",
    ],
    "analysis": [
        "k8s_target_port-misconfig-analysis-1",
        "k8s_target_port-misconfig-analysis-2",
        "k8s_target_port-misconfig-analysis-3",
        "auth_miss_mongodb-analysis-1",
        "revoke_auth_mongodb-analysis-1",
        "revoke_auth_mongodb-analysis-2",
        "user_unregistered_mongodb-analysis-1",
        "user_unregistered_mongodb-analysis-2",
        "misconfig_app_hotel_res-analysis-1",
        "scale_pod_zero_social_net-analysis-1",
        "assign_to_non_existent_node_social_net-analysis-1",
        "redeploy_without_PV-analysis-1",
        "wrong_bin_usage-analysis-1",
    ],
    "mitigation": [
        "k8s_target_port-misconfig-mitigation-1",
        "k8s_target_port-misconfig-mitigation-2",
        "k8s_target_port-misconfig-mitigation-3",
        "auth_miss_mongodb-mitigation-1",
        "revoke_auth_mongodb-mitigation-1",
        "revoke_auth_mongodb-mitigation-2",
        "user_unregistered_mongodb-mitigation-1",
        "user_unregistered_mongodb-mitigation-2",
        "misconfig_app_hotel_res-mitigation-1",
        "scale_pod_zero_social_net-mitigation-1",
        "assign_to_non_existent_node_social_net-mitigation-1",
        "redeploy_without_PV-mitigation-1",
        "wrong_bin_usage-mitigation-1",
    ],
}

class ExperimentRunner:
    def __init__(self, output_dir: str, run_name: str, log_level: str = "INFO"):
        """
        Initialize the experiment runner.

        Args:
            output_dir: Directory to save logs and results
            run_name: Name of the experiment run (for grouping) - REQUIRED
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.run_name = run_name
        self.base_output_dir = Path(output_dir)
        self.output_dir = self.base_output_dir / "run_name" /self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging(log_level)

        # Default configuration
        self.pids = [
            "cpu_stress_hotel_res-localization-1",
            "memory_stress_social_net-localization-1",
            "network_delay_hotel_res-localization-1",
            "scale_pod_zero_social_net-localization-1",
        ]

        self.models = ["qwen3:32b"]
        self.agents = ["kiara_agent"]
        self.free_intv = "5s"
        self.fault_intv = "10s"
        self.runs_per_experiment = 1
        self.max_steps = 30  # default if not set

        # Results storage
        self.results = []

    def setup_logging(self, log_level: str):
        """Setup logging configuration."""
        log_file = self.output_dir / "experiment.log"

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")

    def configure_experiments(self, pids: List[str] = None, pid_type: str = None, models: List[str] = None,
                            agents: List[str] = None, free_intv: str = None,
                            fault_intv: str = None, runs_per_experiment: int = None,
                            max_experiments: int = None, max_steps: int = None):
        """
        Configure experiment parameters.

        Args:
            pids: List of process IDs to test
            pid_type: Type of PIDs to use (detection, localization, analysis, mitigation)
            models: List of models to use
            agents: List of agents to test
            free_intv: Free interval duration
            fault_intv: Fault interval duration
            runs_per_experiment: Number of runs per experiment combination
            max_experiments: Maximum number of experiments to run (only applies when pid_type is specified)
        """
        if pid_type:
            self.pids = PIDS_BY_TYPE.get(pid_type, [])
            if max_experiments is not None:
                # Take the first N PIDs from the registry
                if max_experiments < len(self.pids):
                    self.pids = self.pids[:max_experiments]
                    self.logger.info(f"Limited to first {max_experiments} PIDs from {pid_type} registry")
        if pids is not None:
            self.pids = pids
        if models is not None:
            self.models = models
        if agents is not None:
            self.agents = agents
        if free_intv is not None:
            self.free_intv = free_intv
        if fault_intv is not None:
            self.fault_intv = fault_intv
        if runs_per_experiment is not None:
            self.runs_per_experiment = runs_per_experiment
        if max_steps is not None:
            self.max_steps = max_steps

        self.logger.info(f"Experiment configuration:")
        self.logger.info(f"  PIDs: {self.pids}")
        self.logger.info(f"  Models: {self.models}")
        self.logger.info(f"  Agents: {self.agents}")
        self.logger.info(f"  Free interval: {self.free_intv}")
        self.logger.info(f"  Fault interval: {self.fault_intv}")
        self.logger.info(f"  Max steps: {self.max_steps}")
        self.logger.info(f"  Runs per experiment: {self.runs_per_experiment}")

    def _create_experiment_result(self, experiment_id: str, pid: str, agent_name: str, model: str,
                                run_number: int, start_time: datetime, end_time: datetime,
                                return_code: int, stdout: str, stderr: str,
                                log_file: str, error: str = None) -> Dict[str, Any]:
        """Create a standardized experiment result dictionary."""
        duration = (end_time - start_time).total_seconds()

        result = {
            "experiment_id": experiment_id,
            "run_name": self.run_name,
            "pid": pid,
            "agent": agent_name,
            "model": model,
            "run_number": run_number,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "return_code": return_code,
            "success": return_code == 0,
            "stdout": stdout,
            "stderr": stderr,
            "log_file": str(log_file)
        }

        if error:
            result["error"] = error

        return result

    def run_agent(
        self,
        pid: str,
        agent_name: str,
        model: str,
        max_steps: int,
        run_number: int,
        result_file: str,
    ) -> Dict[str, Any]:
        """
        Run a single agent experiment.

        Args:
            pid: Process ID to test
            agent_name: Name of the agent
            model: Model to use
            run_number: Current run number
            result_file: Optional file name to save results to

        Returns:
            Dictionary containing experiment results
        """
        experiment_id = f"{agent_name}_{model}_{pid}_run_{run_number}"
        aiops_result_file = result_file.with_suffix(".json")
        start_time = datetime.now()

        self.logger.info(f"Starting experiment: {experiment_id}")

        try:
            # Run the agent command
            cmd = [
                "python3",
                "agent_main.py",
                "-p", pid,
                "-a", agent_name,
                "-m", model,
                "-d",
                "--free_intv", self.free_intv,
                "--fault_intv", self.fault_intv,
                "--max-steps", str(max_steps),
                "--result-path", str(aiops_result_file.parent),
                "--result-file", aiops_result_file.name
            ]

            self.logger.debug(f"Running command: {' '.join(cmd)}")

            with open(result_file, 'w') as f:
                f.write(f"Experiment: {experiment_id}\n")
                f.write(f"Run name: {self.run_name}\n")
                f.write(f"Start time: {start_time}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write("\n=== OUTPUT (stdout & stderr) ===\n")
                f.flush()

                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=f,
                    text=True
                )
                process.wait(timeout=25*60) # 25 min

                end_time = datetime.now()

            # Since output is streamed, we can't capture stdout/stderr here
            experiment_result = self._create_experiment_result(
                experiment_id, pid, agent_name, model, run_number,
                start_time, end_time, process.returncode,
                stdout="(see log file)", stderr="(see log file)", log_file=result_file
            )

            if process.returncode == 0:
                self.logger.info(f"Experiment completed successfully in {experiment_result['duration_seconds']:.2f}s")
            else:
                self.logger.warning(f"Experiment failed with return code {process.returncode}")

            return experiment_result

        except subprocess.TimeoutExpired:
            process.kill()
            self.logger.error(f"Experiment timed out after 1 hour")
            return self._create_experiment_result(
                experiment_id, pid, agent_name, model, run_number,
                start_time, datetime.now(), -1, "", "Timeout after 1 hour", result_file, "Timeout"
            )
        except Exception as e:
            self.logger.error(f"Experiment failed with exception: {str(e)}")
            return self._create_experiment_result(
                experiment_id, pid, agent_name, model, run_number,
                start_time, datetime.now(), -1, "", str(e), result_file, str(e)
            )

    def _get_result_paths(self, pid, agent_name, model, run_number):
        """Helper to build result_path and result_file for an experiment."""
        agent_model = f"{agent_name}_{model}"
        result_path = self.base_output_dir / "pid" / pid / self.run_name / agent_model
        result_path.mkdir(parents=True, exist_ok=True)
        result_file = result_path / f"run_{run_number}.log"
        return result_path, result_file

    def run_all_experiments(self):
        """Run all configured experiments."""
        total_experiments = len(self.agents) * len(self.models) * len(self.pids) * self.runs_per_experiment
        current_experiment = 0

        self.logger.info(f"Starting {total_experiments} experiments...")

        for agent_name in self.agents:
            for pid in self.pids:
                for model in self.models:
                    for run_number in range(self.runs_per_experiment):
                        current_experiment += 1
                        result_path, result_file = self._get_result_paths(pid, agent_name, model, run_number)
                        self.logger.info(f"Progress: {current_experiment}/{total_experiments} ({current_experiment/total_experiments*100:.1f}%)")

                        # Run the experiment
                        result = self.run_agent(pid, agent_name, model, self.max_steps, run_number, result_file)

                        self.results.append(result)
                        # Save results after each experiment
                        self.save_results()
                        # Small delay between experiments
                        time.sleep(1)

        self.logger.info("All experiments completed!")
        self.generate_summary()

    def save_results(self):
        """Save current results to JSON file."""
        results_file = self.output_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.logger.debug(f"Results saved to {results_file}")

    def generate_summary(self):
        """Generate a summary of all experiments."""
        if not self.results:
            self.logger.warning("No results to summarize")
            return

        total_experiments = len(self.results)
        successful_experiments = sum(1 for r in self.results if r.get('success', False))
        failed_experiments = total_experiments - successful_experiments

        # Calculate average duration
        durations = [r.get('duration_seconds', 0) for r in self.results if r.get('duration_seconds')]
        avg_duration = sum(durations) / len(durations) if durations else 0

        summary = {
            "run_name": self.run_name,
            "summary": {
                "total_experiments": total_experiments,
                "successful_experiments": successful_experiments,
                "failed_experiments": failed_experiments,
                "success_rate": successful_experiments / total_experiments if total_experiments > 0 else 0,
                "average_duration_seconds": avg_duration,
                "total_duration_seconds": sum(durations)
            },
            "experiments": self.results
        }

        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Summary generated:")
        self.logger.info(f"  Total experiments: {total_experiments}")
        self.logger.info(f"  Successful: {successful_experiments}")
        self.logger.info(f"  Failed: {failed_experiments}")
        self.logger.info(f"  Success rate: {summary['summary']['success_rate']:.2%}")
        self.logger.info(f"  Average duration: {avg_duration:.2f}s")
        self.logger.info(f"  Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Run automated experiments with multiple PIDs")
    parser.add_argument("--output-dir", "-o", default="./experiment_results",
                       help="Output directory for logs and results")
    parser.add_argument("--run-name", "-r", required=True,
                       help="Name for this experiment run (used for grouping and subfolder) - REQUIRED")
    parser.add_argument("--log-level", "-l", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--pid-type", choices=["detection", "localization", "analysis", "mitigation"],
                       help="Type of PIDs to use")
    parser.add_argument("--pids", nargs="+",
                       help="List of PIDs to test (overrides pid-type)")
    parser.add_argument("--max-experiments", type=int, default=None,
                       help="Maximum number of experiments to run (only applies when pid-type is specified)")
    parser.add_argument("--models", nargs="+",
                       help="List of models to use (overrides default)")
    parser.add_argument("--agents", nargs="+",
                       help="List of agents to test (overrides default)")
    parser.add_argument("--free-intv", default="20s",
                       help="Free interval duration")
    parser.add_argument("--fault-intv", default="40s",
                       help="Fault interval duration")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of runs per experiment combination")
    parser.add_argument("--max-steps", default=None,
                       help="Maximum number of steps for each agent run (overrides default)")

    args = parser.parse_args()

    # Create and configure experiment runner
    runner = ExperimentRunner(args.output_dir, args.run_name, args.log_level)

    # Configure experiments with command line arguments
    runner.configure_experiments(
        pids=args.pids,
        models=args.models,
        agents=args.agents,
        free_intv=args.free_intv,
        fault_intv=args.fault_intv,
        runs_per_experiment=args.runs,
        pid_type=args.pid_type,
        max_experiments=args.max_experiments,
        max_steps=args.max_steps
    )

    # Run all experiments
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
