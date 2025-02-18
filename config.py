import dotenv
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
)
import os


dotenv.load_dotenv()

LLM = AzureChatOpenAI(
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    model_name="gpt-4o",  # Specify the base model name
    temperature=0,
)

EMBEDDINGS = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "eagle_openai_embeddings"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    model="text-embedding-ada-002",  # Specify the base model name
    disallowed_special=(),
)
