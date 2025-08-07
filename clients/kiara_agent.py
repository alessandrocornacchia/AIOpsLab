import asyncio
import argparse
import os

from aiopslab.orchestrator import Orchestrator
from clients.utils.llm import Ollama
from clients.utils.templates import DOCS, DOCS_ACTIONS
from clients.utils.llm_tracing import initLangFuse
from aiopslab.paths import SRSI_RESULTS_DIR

from langfuse import observe

class Agent():
    def __init__(self, model_name):
        self.history = []
        self.llm = Ollama(model_name)
        self.improvement_level = os.getenv("IMPROVEMENT_LEVEL")

    def init_context(self, problem_desc: str, instructions: str, apis: str):
        """Initialize the context for the agent."""

        self.shell_api = self._filter_dict(apis, lambda k, _: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)
        self.telemetry_apis = self._filter_dict(
            apis, lambda k, _: "exec_shell" not in k and "submit" not in k
        )

        stringify_apis = lambda apis: "\n\n".join(
            [f"{k}\n{v}" for k, v in apis.items()]
        )

        self.system_message = DOCS.format(
            prob_desc=problem_desc,
            telemetry_apis=stringify_apis(self.telemetry_apis),
            shell_api=stringify_apis(self.shell_api),
            submit_api=stringify_apis(self.submit_api),
        )

        self.available_actions = DOCS_ACTIONS.format(
            telemetry_apis=stringify_apis(self.telemetry_apis),
            shell_api=stringify_apis(self.shell_api),
            submit_api=stringify_apis(self.submit_api),
        )

        self.task_message = instructions

        if self.improvement_level == "remind_actions":
            self.task_message += "Respond only with the action you will take."
        

        self.history.append({"role": "system", "content": self.system_message})
        self.history.append({"role": "user", "content": self.task_message})

    async def get_action(self, input) -> str:
        """Wrapper to interface the agent with OpsBench.

        Args:
            input (str): The input from the orchestrator/environment.

        Returns:
            str: The response from the agent.
        """
        self.history.append({"role": "user", "content": self._add_instr(input)})
#        print(f"Agent response: {self.history}") # debug information
        response = self.llm.run(self.history)
#        print(f"Agent response: {response}") # debug information
        self.history.append({"role": "assistant", "content": response})
        return response

    def _filter_dict(self, dictionary, filter_func):
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}

    def _add_instr(self, input):
        return input


from dotenv import load_dotenv
load_dotenv()

@observe
def main():
    parser=argparse.ArgumentParser(description="Testing, from qwen_copy.py")
    parser.add_argument("--pid", "-p")
    parser.add_argument("--model", "-m")
    parser.add_argument("--delete", "-d", action="store_true")
    parser.add_argument("--free_intv", default="5s")
    parser.add_argument("--fault_intv", default="10s")
    args=parser.parse_args()
    print(args)
    initLangFuse()
    agent = Agent(args.model)

    orchestrator = Orchestrator()
    orchestrator.register_agent(agent, name=f"qwen_copy-{args.model}")

    # pid = "cpu_stress_hotel_res-localization-1"
    # this is the problem ID you want to solve. You can find the problem
    # list in the orchestrator's `problems` directory.

    problem_desc, instructs, apis = orchestrator.init_problem(args.pid, delete_app=args.delete,
                                                            fault_free_interval=args.free_intv, fault_interval=args.fault_intv,
                                                            )
    print(problem_desc, instructs, apis)
    agent.init_context(problem_desc, instructs, apis)
    asyncio.run(orchestrator.start_problem(max_steps=30, path_name=SRSI_RESULTS_DIR/args.pid))

if __name__ == "__main__":
    main()