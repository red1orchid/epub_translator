import abc
from typing import List, Dict


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send messages and return the assistant's reply text."""
        ...


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return completion.choices[0].message.content


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # Anthropic separates system from user/assistant messages
        system_msg = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                chat_messages.append(m)

        kwargs = dict(model=self.model, max_tokens=16384, messages=chat_messages)
        if system_msg:
            kwargs["system"] = system_msg

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


def create_provider(provider_name: str, api_key: str, model: str) -> LLMProvider:
    if provider_name == "openai":
        return OpenAIProvider(api_key=api_key, model=model)
    elif provider_name == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
