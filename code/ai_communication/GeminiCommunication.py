from ai_communication.AICommunication import AICommunication

class GeminiCommunication(AICommunication):
    def __init__(self, client: str, model: str):
        self.client = client
        self.model = model

    def send_message(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )

        return response.output_text