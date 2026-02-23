from ai_communication.AICommunication import AICommunication
from openai import OpenAI

class OpenAICommunication(AICommunication):
    # Ask for the fully built OpenAI client here
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def send_message(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )

        return response.output_text