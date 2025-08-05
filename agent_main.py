import argparse
import asyncio

from clients.utils.llm_tracing import initLangFuse
from aiopslab.orchestrator import Orchestrator
from aiopslab.paths import SRSI_RESULTS_DIR
from langfuse import observe


def get_agent(agent_name, model="qwen3:32b"):
    #qwen_copy needs model_name to be passed
    if agent_name == "kiara_agent":
        from clients.kiara_agent import Agent as KiaraAgent
        agent_class = KiaraAgent

        agent = agent_class(model)
        return agent

    if agent_name == "deepseek":
        from clients.deeepseek import Agent as DeepSeekAgent
        agent_class = DeepSeekAgent
    elif agent_name == "dspy":
        from clients.dspy_agent import ReActAgent as DspyAgent
        agent_class = DspyAgent
    elif agent_name == "flash":
        from clients.flash import FlashAgent
        agent_class = FlashAgent
    elif agent_name == "gpt_managed_identity":
        from clients.gpt_managed_identity import Agent as GPTManagedIdenAgent
        agent_class = GPTManagedIdenAgent
    elif agent_name == "gpt":
        from clients.gpt import Agent as GPTAgent
        agent_class = GPTAgent
    elif agent_name == "langgraph":
        from clients.langgraph_agent import LangGraphAgent
        agent_class = LangGraphAgent
    elif agent_name == "minstral":
        from clients.minstral import Agent as MinstralAgent
        agent_class = MinstralAgent
    elif agent_name == "multi_agent":
        from clients.multi_agent import MultiAgentReAct
        agent_class = MultiAgentReAct
    elif agent_name == "react_4o":
        from clients.react_4o import Agent as ReAct4oAgent
        agent_class = ReAct4oAgent
    elif agent_name == "react_deepseek":
        from clients.react_deepseek import Agent as ReActDeepSeekAgent
        agent_class = ReActDeepSeekAgent
    elif agent_name == "react_o1":
        from clients.react_o1 import Agent as ReActo1Agent
        agent_class = ReActo1Agent
    else:
        raise ValueError(f"Invalid option: {agent_name}")

    agent = agent_class()
    return agent

def parse_arguments():
    parser=argparse.ArgumentParser(description="Testing, from qwen_copy.py")
    parser.add_argument("--pid", "-p")
    parser.add_argument("--agent_name", "-a")
    parser.add_argument("--model", "-m")
    parser.add_argument("--delete", "-d", action="store_true")
    parser.add_argument("--free_intv", default="5s")
    parser.add_argument("--fault_intv", default="10s")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum number of steps for the orchestrator")
    parser.add_argument("--result-path", type=str, default=None, help="Custom path for results/logs for this run")
    parser.add_argument("--result-file", type=str, default=None, help="Path to a result file for this run (optional)")
    args=parser.parse_args()
    return args

from dotenv import load_dotenv
load_dotenv()

@observe
def main():
    args = parse_arguments()
    initLangFuse()
    agent = get_agent(args.agent_name, args.model)

    orchestrator = Orchestrator()
    orchestrator.register_agent(agent, name=f"{args.agent_name}-{args.model}")

    problem_desc, instructs, apis = orchestrator.init_problem(args.pid, delete_app=args.delete,
                                                            fault_free_interval=args.free_intv, fault_interval=args.fault_intv,
                                                            )
    print(problem_desc, instructs, apis)
    agent.init_context(problem_desc, instructs, apis)
    output_path = args.result_path if args.result_path else SRSI_RESULTS_DIR/args.pid
    asyncio.run(orchestrator.start_problem(
        max_steps=args.max_steps,
        result_path=output_path,
        result_file=args.result_file
    ))

if __name__ == "__main__":
    main()
