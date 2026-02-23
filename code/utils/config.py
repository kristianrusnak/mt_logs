from ai_factory.OpenAIFactory import OpenAIFactory
from openai import OpenAI
import os
from dotenv import load_dotenv



def get_client():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    openai_client = OpenAI()
    openai_client.api_key = openai_api_key
    client = OpenAIFactory(
        client=openai_client,
        model="gpt-5-mini"
    )
    return client