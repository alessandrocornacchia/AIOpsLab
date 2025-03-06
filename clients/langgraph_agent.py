import asyncio
from langgraph.graph import StateGraph
from clients.utils.llm import GPT4Turbo
from clients.utils.templates import DOCS_SHELL_ONLY

class AgentState:
    def __init__(self, history: list[dict[str, str]], problem_desc: str, instructions: str, shell_api: dict, submit_api: dict):
        self.history = history
        self.problem_desc = problem_desc
        self.instructions = instructions
        self.shell_api = shell_api
        self.submit_api = submit_api

class LangGraphAgent:
    def __init__(self):
        self.llm = GPT4Turbo()
        self.graph = StateGraph(AgentState)
        self.agent_state = None
        self._build_graph()

    def _build_graph(self):
        self.graph.add_node("process_input", self.process_input)
        self.graph.add_node("generate_response", self.generate_response)
        self.graph.set_entry_point("process_input")
        self.graph.add_edge("process_input", "generate_response")
        self.processor = self.graph.compile()

    def init_context(self, problem_desc: str, instructions: str, apis: dict):
        """Initialize the context for the agent."""
        shell_api = self._filter_dict(apis, lambda k, _: "exec_shell" in k)
        submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)
        stringify_apis = lambda apis: "\n\n".join([f"{k}\n{v}" for k, v in apis.items()])
        
        system_message = DOCS_SHELL_ONLY.format(
            prob_desc=problem_desc,
            shell_api=stringify_apis(shell_api),
            submit_api=stringify_apis(submit_api),
        )

        self.agent_state =  AgentState(
            history=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": instructions}
            ],
            problem_desc=problem_desc,
            instructions=instructions,
            shell_api=shell_api,
            submit_api=submit_api,
        )
        return self.agent_state
    
    async def process_input(self, state: AgentState, user_input: str):
        """Handles user input and updates history."""
        state.history.append({"role": "user", "content": user_input})
        return state
    
    async def generate_response(self, state: AgentState):
        """Generates response using the LLM."""
        response = self.llm.run(state.history)
        state.history.append({"role": "assistant", "content": response[0]})
        return state, response[0]
    
    async def get_action(self, input: str) -> str:
        """Wrapper to interface the agent for processing input and generating a response."""
        state = await self.process_input(self.agent_state, input)
        state, response = await self.generate_response(state)
        return response

    def _filter_dict(self, dictionary, filter_func):
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}

if __name__ == "__main__":
    agent = LangGraphAgent()
    
    from aiopslab.orchestrator import Orchestrator
    orchestrator = Orchestrator()
    orchestrator.register_agent(agent, name="langgraph-gpt-agent")
    
    pid = "misconfig_app_hotel_res-detection-1"
    problem_desc, instructs, apis = orchestrator.init_problem(pid)
    initial_state = agent.init_context(problem_desc, instructs, apis)
    
    asyncio.run(orchestrator.start_problem(max_steps=10))
