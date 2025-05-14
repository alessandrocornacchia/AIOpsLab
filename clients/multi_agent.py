import os
import asyncio
import autogen

from aiopslab.orchestrator import Orchestrator
from clients.utils.templates import AUTOGEN_DOCS

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
        llm_config={"model": "gpt-4o", "api_key": api_key}
        # Define Autogen Agents
        # self.reasoner = autogen.AssistantAgent(
        #     name="Reasoner",
        #     system_message="You analyze outputs of the given analysis tools to determine what the next API call should be. You only have 30 steps, make sure they count.",
        #     llm_config={"model": "o1", "api_key": api_key}
        # )

        # self.critic = autogen.AssistantAgent(
        #     name="Critic",
        #     system_message="You may only say whether we have analyzed enough for submission or if more analysis is required. Do not ask the user for input.",
        #     llm_config={"model": "gpt-4o", "api_key": api_key}
        # )

        # self.executor = autogen.AssistantAgent(
        #     name="Executor",
        #     system_message="You must respond with the API call that the Reasoner provides inside a markdown code block preceded outside the block by the text 'Action:' , e.g. Action: ```\n<API_NAME>(<API_PARAM1>, <API_PARAM2> ...)\n```.",
        #     llm_config={"model": "gpt-4o", "api_key": api_key}
        # )

        self.reasoner = autogen.AssistantAgent(
            name="Reasoner",
            system_message="You analyze outputs of the given analysis tools to determine what the next API call should be.",
            llm_config=llm_config
        )

        self.critic = autogen.AssistantAgent(
            name="Critic",
            system_message="You evaluate the actions taken. Ensure that proper analysis is done before confirming anomalies.",
            llm_config=llm_config
        )

        self.executor = autogen.AssistantAgent(
            name="Executor",
            system_message="You must respond with exactly one API call that the Reasoner provides inside a markdown code block preceded outside the block by the text 'Action:' , e.g. Action: ```\n<API_NAME>(<API_PARAM1>, <API_PARAM2> ...)\n```. Do not explain or add extra text.",
            llm_config=llm_config
        )

        # Define Multi-Agent GroupChat with Custom Speaker Selection
        self.groupchat = autogen.GroupChat(
            agents=[self.critic, self.reasoner, self.executor],
            messages=[],
            max_round=20
        )

        # Define Multi-Agent Collaboration
        self.team = autogen.GroupChatManager(
            groupchat=self.groupchat,
            llm_config=llm_config,
            is_termination_msg=lambda msg: "action:" in msg["content"].lower()
        )

    def init_context(self, problem_desc: str, instructions: str, apis: dict):
        """Initialize problem-solving context for agents."""
        self.problem_desc = problem_desc
        self.instructions = instructions
        self.shell_api = self._filter_dict(apis, lambda k, _: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)
        self.telemetry_apis = self._filter_dict(
            apis, lambda k, _: "exec_shell" not in k and "submit" not in k
        )

        # Format API Details
        stringify_apis = lambda apis: "\n\n".join([f"{k}\n{v}" for k, v in apis.items()])
        
        self.system_message = AUTOGEN_DOCS.format(
            prob_desc=problem_desc,
            telemetry_apis=stringify_apis(apis),
            shell_api=stringify_apis(self.shell_api),
            submit_api=stringify_apis(self.submit_api)
        )

        self.task_message = instructions

        self.history.append({"role": "system", "content": self.system_message})
        self.history.append({"role": "user", "content": self.task_message})

    def _filter_dict(self, dictionary, filter_func):
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}
    
    def format_conversation_autogen(self, messages):
        """Format multi-agent messages into a readable conversation."""
        return "\n\n".join(
            f"{msg['name']}:\n{msg['content'].strip()}"
            for msg in messages[1:]  # skip the first user message
            if msg["role"] == "assistant"
        )
    
    def format_conversation(self, messages):
        return "\n\n".join(
            f"{msg['role']}:\n{msg['content'].strip()}"
            for msg in messages
        )
    def _ensure_markdown_format(self, text: str) -> str:
        import re

        # Find all markdown code blocks and reformat them to just ```\n...\n```
        def format_block(match):
            code = match.group(1).strip()
            return f"\n```\n{code}\n```\n"

        # This handles both ```lang\ncode``` and ```code```
        return re.sub(r'```(?:\w*\n)?(.*?)```', format_block, text, flags=re.DOTALL).strip()


    async def get_action(self, input_text: str) -> str:
        """Generate the agent's action using Autogen multi-agent reasoning."""
        self.history.append({"role": "user", "content": input_text})
        conversation = self.format_conversation(self.history)
        # Start Multi-Agent Conversation
        response = self.team.initiate_chat(
            self.reasoner,
            message={"role": "user", "content": conversation}
        )
        response = self.format_conversation_autogen(response.chat_history)
        response = self._ensure_markdown_format(response)
        # Ensure response is a string to avoid validation errors
        self.history.append({"role": "assistant", "content": str(response)})
        return str(response)


if __name__ == "__main__":
    for i in range(1):
        print(f"Running AutoGen - 3 gpt4o - with new tools: {i}")
        agent = MultiAgentReAct()

        orchestrator = Orchestrator()
        orchestrator.register_agent(agent, name="AutoGen - 3 gpt4o - with new tools")

        pid = "assign_to_non_existent_node_social_net-localization-1"
        fault_free_interval = '60s'
        fault_interval = '60s'
        num_failures = 1
        problem_desc, instructs, apis = orchestrator.init_problem(pid, fault_free_interval, fault_interval, num_failures)
        agent.init_context(problem_desc, instructs, apis)

        asyncio.run(orchestrator.start_problem(max_steps=30))
