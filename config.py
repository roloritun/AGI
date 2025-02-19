import dotenv
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
)
from langchain_core.callbacks import CallbackManager
import os
from utils.performance import PerformanceCallbackHandler, PerformanceStats

dotenv.load_dotenv()

# Initialize performance tracking
performance_stats = PerformanceStats()
callback_handler = PerformanceCallbackHandler(performance_stats)
callback_manager = CallbackManager([callback_handler])

LLM = AzureChatOpenAI(
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    model_name="gpt-4",
    temperature=0,
    callback_manager=callback_manager,
    streaming=True,
)

EMBEDDINGS = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "eagle_openai_embeddings"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    model="text-embedding-ada-002",
    disallowed_special=(),
)

# Export performance stats for use in other modules
__all__ = ["LLM", "EMBEDDINGS", "performance_stats"]
