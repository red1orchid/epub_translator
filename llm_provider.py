import abc
from typing import List, Dict

__all__ = ["LLMProvider", "OpenAIProvider", "AnthropicProvider", "create_provider"]


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
            # gpt-5.x and o-series reject the legacy max_tokens parameter
            max_completion_tokens=16384,
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
        # Newer models (e.g. claude-opus-5) think by default: content may start
        # with thinking blocks, so collect text blocks instead of content[0]
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise RuntimeError(
                f"LLM response contains no text (stop_reason={response.stop_reason}, "
                f"blocks: {[block.type for block in response.content]})"
            )
        return text


def create_provider(provider_name: str, api_key: str, model: str) -> LLMProvider:
    if provider_name == "openai":
        return OpenAIProvider(api_key=api_key, model=model)
    elif provider_name == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
