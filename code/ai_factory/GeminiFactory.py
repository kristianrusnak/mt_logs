from ai_factory import AIFactory
from ai_communication import AICommunication

class GeminiFactory(AIFactory):
    def __init__(self, client: str, model: str):
        self.model = model
        self.client = client

    def create_communication(self) -> AICommunication:
        raise NotImplementedError("Gemini communication module is not implemented yet.")
        pass