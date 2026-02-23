from ai_factory.AIFactory import AIFactory
from ai_communication.AICommunication import AICommunication
from ai_communication.OpenAICommunication import OpenAICommunication
from openai import OpenAI

class OpenAIFactory(AIFactory):
    def __init__(self, client: OpenAI, model: str):
        self.model = model
        self.client = client

    def create_communication(self) -> AICommunication:
        return OpenAICommunication(
            client=self.client, 
            model=self.model
        )