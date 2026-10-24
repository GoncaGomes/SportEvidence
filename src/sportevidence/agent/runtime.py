"""Build the SportEvidence agent."""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from sportevidence.agent.model import create_chat_model
from sportevidence.agent.prompts import SYSTEM_PROMPT
from sportevidence.agent.tools import create_query_tool


def create_evidence_agent(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    return create_agent(
        model=create_chat_model(),
        tools=[create_query_tool()],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def main() -> int:
    load_dotenv()

    llm = create_chat_model()
    search_tool = create_query_tool()

    agent = create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt=SYSTEM_PROMPT,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Search running economy and resistance training just reply with the first evidence search, then summrize it.",
                }
            ]
        },
        config={"recursion_limit": 10},
    )

    for message in result["messages"]:
        print(f"\n---{message.type}---")

        if getattr(message, "tool_calls", None):
            print("Tool calls", message.tool_calls)

        print(message.content)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
