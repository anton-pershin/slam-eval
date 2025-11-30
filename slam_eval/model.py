from abc import ABC, abstractmethod
from typing import Any

from rally.interaction import request_based_on_message_history
from rally.llm import Llm

from slam_eval.collections.text_generation import TextGenerationInput


class Model(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def predict(self, x: Any) -> Any: ...


class LlmViaOpenAiApi(Model):
    def __init__(self, name: str, llm: Llm) -> None:
        super().__init__(name)
        self.llm = llm

    def predict(self, x: TextGenerationInput) -> str:
        messages = []

        if x["system_prompt"] is not None:
            messages.append(
                {
                    "role": "system",
                    "content": x["system_prompt"],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": x["user_prompt"],
            }
        )

        resp_message = request_based_on_message_history(
            llm_server_url=self.llm.url,
            message_history=messages,
            authorization=self.llm.authorization,
            model=self.llm.model,
            max_output_tokens=self.llm.max_output_tokens,
        )

        return resp_message["content"]
