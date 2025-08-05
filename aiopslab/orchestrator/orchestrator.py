# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Orchestrator class that interfaces with the agent and the environment."""

from aiopslab.service.helm import Helm
from aiopslab.service.kubectl import KubeCtl
from aiopslab.session import Session
from aiopslab.orchestrator.problems.registry import ProblemRegistry
from aiopslab.orchestrator.parser import ResponseParser
from aiopslab.utils.status import *
from aiopslab.service.telemetry.prometheus import Prometheus
import time
import inspect
import asyncio
import re
from aiopslab.paths import SRSI_RESULTS_DIR

class Orchestrator:
    def __init__(self):
        self.agent = None
        self.session = None
        self.parser = ResponseParser()
        self.probs = ProblemRegistry()
        self.sprint = SessionPrint()
        self.execution_start_time = None
        self.execution_end_time = None
        self.kubectl = KubeCtl()


    def init_problem(self, problem_id: str, delete_app: bool, fault_free_interval: str = "60s", fault_interval: str = "60s", num_failures: int = 1):
        """Initialize a problem instance for the agent to solve.

        Args:
            problem_id (str): The problem instance identifier.

        Returns:
            tuple: A tuple containing the problem description, task message, and session object.
        """
        # Start timer
        self.execution_start_time = time.time()

        self.session = Session()
        print(f"Session ID: {self.session.session_id}")
        prob = self.probs.get_problem_instance(problem_id)
        self.session.set_problem(prob, pid=problem_id)
        self.session.set_agent(self.agent_name)

        print("Setting up OpenEBS...")

        command = "kubectl get pods -n openebs"
        result = self.kubectl.exec_command(command)
        if "Running" in result:
            print("OpenEBS is already running. Skipping installation.")
        else:
            self.kubectl.exec_command(
                "kubectl apply -f https://openebs.github.io/charts/openebs-operator.yaml"
            )
            self.kubectl.exec_command(
                "kubectl patch storageclass openebs-hostpath -p '{\"metadata\": {\"annotations\":{\"storageclass.kubernetes.io/is-default-class\":\"true\"}}}'"
            )
            self.kubectl.wait_for_ready("openebs")
            print("OpenEBS setup completed.")

        # Setup and deploy Prometheus
        self.prometheus = Prometheus()
        self.prometheus.deploy()

        # deploy service, if needed:
        if delete_app:
            prob.app.delete()

        prob.app.deploy()

        if 'cpu_stress' in problem_id or 'memory_stress' in problem_id or 'network_delay' in problem_id:

            int_fault_free_interval = self.parse_duration(fault_free_interval)
            int_fault_interval = self.parse_duration(fault_interval)
            total_duration = (int_fault_free_interval + int_fault_interval) * num_failures

            if inspect.iscoroutinefunction(prob.start_workload):
                asyncio.create_task(prob.start_workload(total_duration))
            else:
                prob.start_workload(total_duration)

            for i in range(num_failures):
                print(f"Fault injection {i+1} of {num_failures}, sleeping for {int_fault_free_interval} seconds for no faults...")
                time.sleep(int_fault_free_interval)
                print(f"Injecting fault for {int_fault_interval} seconds...")
                prob.inject_fault(fault_interval)
                time.sleep(int_fault_interval)
                prob.recover_fault()

        else:
            # inject fault
            prob.inject_fault()

            # Check if start_workload is async or sync
            if inspect.iscoroutinefunction(prob.start_workload):
                asyncio.create_task(prob.start_workload())
            else:
                prob.start_workload()

        task_desc = prob.get_task_description()
        instructions = prob.get_instructions()
        actions = prob.get_available_actions()

        return task_desc, instructions, actions

    def register_agent(self, agent, name="agent"):
        """Register the agent for the current session.

        Args:
            agent: The agent to register.
            name: The name of the agent (default: "agent").
        """
        self.agent = agent
        self.agent_name = name

    async def ask_agent(self, input):
        """Ask the agent for the next action given the current context."""
        assert self.session is not None
        assert self.agent is not None

        agent_response = await self.agent.get_action(input)
        self.session.add({"role": "assistant", "content": agent_response})

        return agent_response

    async def ask_env(self, input):
        """Ask the environment for the observation given the current action."""
        assert self.session is not None

        try:
            resp = self.parser.parse(input)
        except ResponseParsingError as e:
            self.session.add({"role": "env", "content": str(e)})
            return str(e)

        api, args, kwargs = resp["api_name"], resp["args"], resp["kwargs"]

        # if submit, save solution for eval
        if api == "submit":
            self.session.set_solution(args[0] if len(args) == 1 else args)

        try:
            env_response = self.session.problem.perform_action(api, *args, **kwargs)
        except InvalidActionError as e:
            env_response = str(e)

        self.session.add({"role": "env", "content": env_response})

        return env_response

    async def start_problem(self, max_steps: int, result_path: str = None, result_file: str = None):
        """Start the task and run for a specified number of steps.

        Args:
            max_steps (int): The maximum number of steps to run the task.
            result_path (str): Directory path to save the session JSON.
            result_file (str, optional): Filename for the session JSON. If not provided, a default is used.

        Returns:
            dict: The final state of the session.
        """
        assert self.session is not None
        action_instr = "Please take the next action"
        action, env_response, results = "", "", {}
        self.session.start()

        for step in range(max_steps):
            action = await self.ask_agent(action_instr)
            self.sprint.agent(action)

            env_response = await self.ask_env(action)
            self.sprint.service(env_response)

            if env_response == SubmissionStatus.VALID_SUBMISSION:
                break
            elif env_response == SubmissionStatus.INVALID_SUBMISSION:
                raise ValueError("Invalid submission!")  # TODO (@manish): ask to retry?

            action_instr = env_response + "\n" + "Please take the next action"

        self.session.end()

        # A valid submission was made (or) max_steps reached
        if env_response != SubmissionStatus.INVALID_SUBMISSION:
            results = self.session.problem.eval(
                self.session.solution, self.session.history, self.session.get_duration()
            )
            self.sprint.result(results)

        self.session.set_results(results)
        self.session.to_json(result_path, result_file)
        self.session.problem.recover_fault()

        # Beyond recovering from fault,
        # I feel sometimes it is safer to delete the whole namespace.
        # But this will take more time.
        # if not self.session.problem.sys_status_after_recovery():
        self.session.problem.app.cleanup()

        self.execution_end_time = time.time()
        total_execution_time = self.execution_end_time - self.execution_start_time
        time_keys = ["TTD", "TTL", "TTA", "TTM"]
        key = next((k for k in time_keys if k in results), None)
        framework_overhead = (
            total_execution_time - results[key]
        )  # Time spent doing everything besides running the agent
        print(f"Framework overhead: {framework_overhead}")

        return {
            "history": self.session.history,
            "final_state": env_response,
            "results": results,
            "framework_overhead": framework_overhead,
        }
    def parse_duration(self, duration: str) -> int:
        """
        Converts a duration string (e.g., '5m', '30s', '2h') into seconds.

        Args:
            duration (str): The duration string (e.g., '5m', '30s', '2h').

        Returns:
            int: Duration in seconds.
        """
        match = re.match(r"(\d+)([smh])", duration.lower())
        if not match:
            raise ValueError("Invalid duration format. Use 'Xs', 'Xm', or 'Xh' (e.g., '30s', '5m', '2h').")

        value, unit = int(match.group(1)), match.group(2)

        conversion = {"s": 1, "m": 60, "h": 3600}

        return value * conversion[unit]