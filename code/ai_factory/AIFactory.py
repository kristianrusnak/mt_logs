from abc import ABC, abstractmethod
from ai_communication import AICommunication

class AIFactory(ABC):
    
    @abstractmethod
    def create_communication(self) -> AICommunication:
        pass

    def send_message(self, prompt: str) -> str:
        communication = self.create_communication()
        return communication.send_message(prompt)