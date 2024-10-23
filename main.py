from agents.assistant_agent import AssistantAgent
from agents.reasoning_agent import ReasoningAgent
from agents.research_agent import ResearchAgent
from agent_manager import AgentManager

if __name__ == "__main__":
    manager = AgentManager()
    reasoning_agent = ReasoningAgent()
    research_agent = ResearchAgent()
    assistant_agent = AssistantAgent()

    # Register agents in the order they should process input
    manager.register_agent(reasoning_agent)
    manager.register_agent(research_agent)
    manager.register_agent(assistant_agent)

    manager.interact()