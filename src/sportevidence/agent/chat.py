"""Run an iterative SportEcvidence agent."""

from uuid import uuid4

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

from sportevidence.agent.runtime import create_evidence_agent


def main() -> int:
    load_dotenv()

    checkpointer = InMemorySaver()
    agent = create_evidence_agent(checkpointer=checkpointer)
    thread_id = str(uuid4())

    print("SportEvidence agent running...")
    print("Commands: /new starts a conversation; /exit quites.\n")

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_text:
            continue

        if user_text == "/exit":
            return 0

        if user_text == "/new":
            thread_id = str(uuid4())
            print("Started a new conversation.\n")
            continue

        if user_text not in ["/new", "/exit"]:
            print("Available commands: /new starts a conversation; /exit quites.\n")
            continue

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_text,
                        "thread_id": thread_id,
                    },
                ]
            },
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
        )

        final_message = result["messages"][-1]
        print(f"SportEvidence> {final_message.text}\n")


if __name__ == "__main__":
    raise SystemExit(main())
