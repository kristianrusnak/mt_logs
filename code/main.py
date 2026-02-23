from ai_factory.AIFactory import AIFactory
from utils.config import get_client

if __name__ == "__main__":
    client = get_client()

    message = "What is the capital of France?"
    response = client.send_message(message)
    print(f"\n\nQuestion: {message}\nResponse: {response}\n\n")