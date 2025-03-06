import os
import asyncio
import autogen

from aiopslab.orchestrator import Orchestrator
from clients.utils.templates import DOCS

RESP_INSTR = """DO NOT REPEAT ACTIONS! Respond with:
Thought: <your thought on the previous output>
Action: <your action towards mitigating>
"""

class MultiAgentReAct:
    def __init__(self):
        """Initialize the Multi-Agent ReAct system."""
        self.history = []
        self.problem_desc = ""
        self.instructions = ""
        self.apis = {}

        # Load OpenAI API Key
        api_key = os.getenv("OPENAI_API_KEY")

        # Define Autogen Agents
        self.planner = autogen.AssistantAgent(
            name="Planner",
            system_message="You analyze problems and break them down into actionable steps.",
            llm_config={"model": "gpt-4o-mini", "api_key": api_key}
        )

        self.reasoner = autogen.AssistantAgent(
            name="Reasoner",
            system_message="You analyze logs and metrics to determine anomalies before execution.",
            llm_config={"model": "gpt-4o-mini", "api_key": api_key}
        )

        self.critic = autogen.AssistantAgent(
            name="Critic",
            system_message="You evaluate the actions taken and suggest improvements. Ensure that proper analysis is done before confirming anomalies.",
            llm_config={"model": "gpt-4o-mini", "api_key": api_key}
        )

        self.executor = autogen.AssistantAgent(
            name="Executor",
            system_message="You submit the final API call to the orchestrator.",
            llm_config={"model": "gpt-4o-mini", "api_key": api_key}
        )

        # Define Multi-Agent GroupChat with Custom Speaker Selection
        self.groupchat = autogen.GroupChat(
            agents=[self.planner, self.reasoner, self.critic, self.executor],
            messages=[],
            max_round=15,
            send_introductions=True,
            speaker_selection_method=self.speaker_selection  # Custom speaker selection
        )

        # Define Multi-Agent Collaboration
        self.team = autogen.GroupChatManager(
            groupchat=self.groupchat
        )

    def speaker_selection(self, last_speaker, groupchat):
        """
        Custom speaker selection function ensuring proper agent flow.
        - Planner starts
        - Reasoner analyzes logs/metrics
        - Critic validates analysis
        - Executor submits final action
        """

        messages = groupchat.messages

        if len(messages) <= 1:
            return self.planner  # Start with Planner

        if last_speaker == self.planner:
            return self.reasoner  # Let Reasoner analyze logs & metrics

        if last_speaker == self.reasoner:
            return self.critic  # Critic evaluates the analysis

        if last_speaker == self.critic:
            return self.executor  # Executor submits the final API call

        return "round_robin"  # Default fallback

    def init_context(self, problem_desc: str, instructions: str, apis: dict):
        """Initialize problem-solving context for agents."""
        self.problem_desc = problem_desc
        self.instructions = instructions
        self.apis = apis

        # Format API Details
        stringify_apis = lambda apis: "\n\n".join([f"{k}\n{v}" for k, v in apis.items()])
        
        self.system_message = DOCS.format(
            prob_desc=problem_desc,
            telemetry_apis=stringify_apis(apis),
            shell_api="Handled by Executor",
            submit_api="Handled by Executor"
        )

        # Provide context to the reasoning agent
        self.executor.update_system_message(self.system_message)

    async def get_action(self, input_text: str) -> str:
        """Generate the agent's action using Autogen multi-agent reasoning."""
        self.history.append({"role": "user", "content": input_text + "\n\n" + RESP_INSTR})

        user_message = f"Problem: {self.problem_desc}\nInstructions: {self.instructions}\nObservation: {input_text}"
        user_message += '\nPLEASE COLLECTIVELY DECIDE WHAT API CALL TO RUN NEXT.'
        # Start Multi-Agent Conversation
        response = self.team.run(user_message)

        # Ensure response is a string to avoid validation errors
        self.history.append({"role": "assistant", "content": str(response)})
        return str(response)


if __name__ == "__main__":
    agent = MultiAgentReAct()

    orchestrator = Orchestrator()
    orchestrator.register_agent(agent, name="multiagent_react")

    pid = "k8s_target_port-misconfig-detection-1"
    problem_desc, instructs, apis = orchestrator.init_problem(pid)
    agent.init_context(problem_desc, instructs, apis)

    asyncio.run(orchestrator.start_problem(max_steps=10))
