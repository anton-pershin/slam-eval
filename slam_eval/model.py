from abc import ABC, abstractmethod
from typing import Any

from rally.interaction import request_based_on_message_history
from rally.llm import Llm


class Model(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def predict(self, x: Any) -> Any: ...


class LlmViaOpenAiApi(Model):
    def __init__(self, name: str, llm: Llm) -> None:
        super().__init__(name)
        self.llm = llm

    def predict(self, x: Any) -> Any:
        messages = [{"role": "user", "content": x}]

        resp_message = request_based_on_message_history(
            llm_server_url=self.llm.url,
            message_history=messages,
            authorization=self.llm.authorization,
            model=self.llm.model,
        )

        return resp_message["content"]
