"""Create the chat model used by SportEvidence."""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from sportevidence.config import AgentSettings


def create_chat_model(
    settings: AgentSettings | None = None,
) -> ChatOpenAI:
    """Create a configured chat model"""
    settings = settings if settings is not None else AgentSettings.from_env()

    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_s,
        temperature=0,
    )


def main() -> int:
    """Run a small connectivity check."""
    load_dotenv()

    llm = create_chat_model()

    response = llm.invoke(
        [
            ("system", "Answer briefly."),
            ("human", "Reply with exactly: Connection successful"),
        ]
    )

    print(response.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
