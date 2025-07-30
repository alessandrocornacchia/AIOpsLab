import asyncio
import argparse

from aiopslab.orchestrator import Orchestrator
from clients.utils.llm import Ollama
from clients.utils.templates import DOCS
from clients.utils.llm_tracing import initLangFuse

from langfuse import observe

class Agent():
    def __init__(self):
        self.history = []
        self.llm = Ollama()

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

        self.task_message = instructions

        self.history.append({"role": "system", "content": self.system_message})
        self.history.append({"role": "user", "content": self.task_message})

    async def get_action(self, input, model_name) -> str:
        """Wrapper to interface the agent with OpsBench.

        Args:
            input (str): The input from the orchestrator/environment.

        Returns:
            str: The response from the agent.
        """
        self.history.append({"role": "user", "content": self._add_instr(input)})
        print(f"Agent response: {self.history}") # debug information
        response = self.llm.run(self.history, model_name)
        print(f"Agent response: {response}") # debug information
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
    parser.add_argument("--delete", nargs="?", type=bool, default=False)
    parser.add_argument("--deploy", nargs="?", type=bool, default=False)
    args=parser.parse_args()

    pid = args.pid
    model_name = args.model

    initLangFuse()
    agent = Agent()

    orchestrator = Orchestrator()
    orchestrator.register_agent(agent, name=f"qwen_copy-{model_name}")

    # pid = "astronomy_shop_payment_service_unreachable-localization-1"
    # this is the problem ID you want to solve. You can find the problem
    # list in the orchestrator's `problems` directory.

    problem_desc, instructs, apis = orchestrator.init_problem(pid,
                                                            fault_free_interval="5s", fault_interval="10s",
                                                            delete_service=args.delete, deploy_service=args.deploy)
    print(problem_desc, instructs, apis)
    agent.init_context(problem_desc, instructs, apis)
    asyncio.run(orchestrator.start_problem(max_steps=10, model_name=model_name))
    orchestrator.session.to_json()

if __name__ == "__main__":
    main()