from unittest.mock import Mock

import pytest
from langchain_openai import ChatOpenAI

from sportevidence.agent import model as agent_model
from sportevidence.config import AgentSettings


def test_create_chat_model_uses_explicit_settings() -> None:
    settings = AgentSettings(
        model="gpt-test",
        api_key="key-123",
        base_url="http://localhost:8080/v1",
        timeout_s=12.5,
    )

    llm = agent_model.create_chat_model(settings)

    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-test"
    assert llm.openai_api_key.get_secret_value() == "key-123"
    assert llm.openai_api_base == "http://localhost:8080/v1"
    assert llm.request_timeout == 12.5
    assert llm.temperature == 0


def test_create_chat_model_falls_back_to_environment_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = AgentSettings(model="env-model", api_key="env-key")
    monkeypatch.setattr(
        agent_model, "AgentSettings", Mock(from_env=Mock(return_value=fixed))
    )

    llm = agent_model.create_chat_model()

    assert llm.model_name == "env-model"
    assert llm.openai_api_key.get_secret_value() == "env-key"
