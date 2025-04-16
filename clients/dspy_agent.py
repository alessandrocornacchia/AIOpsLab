import asyncio
import dspy
from aiopslab.orchestrator import Orchestrator
from clients.utils.llm import GPT4Turbo
from clients.utils.templates import DOCS
import os

# Response Instruction to avoid looping actions
RESP_INSTR = """DO NOT REPEAT ACTIONS! Respond with:
Thought: <your thought on the previous output>
Action: <your action towards mitigating IN A MARKDOWN CODE BLOCK>
Remember you are trying to find the root cause of the issue.
"""

# 1. Define a DSPy Signature for Prediction
class ReActSignature(dspy.Signature):
    """Defines structured input/output for DSPy inference."""
    history = dspy.InputField(desc="Past interactions with the agent.")
    observation = dspy.InputField(desc="Latest input from the environment.")
    thought = dspy.OutputField(desc="Agent's reasoning about the situation.")
    action = dspy.OutputField(desc="The corrective action to be taken.")

class ReActAgent:
    def __init__(self):
        self.history = []
        self.llm = dspy.LM(model="openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
        dspy.configure(lm = self.llm)
        self.predictor = dspy.Predict(ReActSignature)

    def init_context(self, problem_desc: str, instructions: str, apis: dict):
        """Initialize agent's problem-solving context."""
        self.shell_api = self._filter_dict(apis, lambda k, _: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)
        self.telemetry_apis = self._filter_dict(
            apis, lambda k, _: "exec_shell" not in k and "submit" not in k
        )

        stringify_apis = lambda apis: "\n\n".join([f"{k}\n{v}" for k, v in apis.items()])
        
        self.system_message = DOCS.format(
            prob_desc=problem_desc,
            telemetry_apis=stringify_apis(self.telemetry_apis),
            shell_api=stringify_apis(self.shell_api),
            submit_api=stringify_apis(self.submit_api),
        )

        self.task_message = instructions

        self.history.append({"role": "system", "content": self.system_message})
        self.history.append({"role": "user", "content": self.task_message})

    def _filter_dict(self, dictionary, filter_func):
        """Helper function to filter dictionary keys."""
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}

    def _add_instr(self, input):
        """Format input with response instructions."""
        return input + "\n\n" + RESP_INSTR

    async def get_action(self, input_text: str) -> str:
        """Generate the agent's action using DSPy inference."""
        self.history.append({"role": "user", "content": self._add_instr(input_text)})
        
        observation = {"history": str(self.history), "observation": input_text}
        
        response = self.predictor.forward(**observation)

        thought, action = response.thought, response.action
        formatted_response = f"Thought: {thought}\nAction: ```\n {action}\n ```"
        
        self.history.append({"role": "assistant", "content": formatted_response})
        return formatted_response


if __name__ == "__main__":
    agent = ReActAgent()

    orchestrator = Orchestrator()
    orchestrator.register_agent(agent, name="dspy_react")

    pid = "cpu_stress_hotel_res-localization-1"
    fault_free_interval = '60s'
    fault_interval = '60s'
    num_failures = 1
    problem_desc, instructs, apis = orchestrator.init_problem(pid, fault_free_interval, fault_interval, num_failures)
    
    agent.init_context(problem_desc, instructs, apis)
    
    asyncio.run(orchestrator.start_problem(max_steps=15))
