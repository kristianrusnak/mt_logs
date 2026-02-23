from abc import ABC, abstractmethod

class AICommunication(ABC):
    @abstractmethod
    def send_message(self, prompt: str) -> str:
        """Sends a message to the AI and returns the response."""
        pass