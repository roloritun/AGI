"""AutoGPT Implementation

This script implements an AutoGPT agent using LangChain primitives
(LLMs, PromptTemplates, VectorStores, Embeddings, Tools)
"""

import os
from typing import Optional
from dotenv import load_dotenv

from langchain.agents import Tool
from langchain_community.tools import OpenWeatherMapQueryRun
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import FileChatMessageHistory

from autogpt.agent import AutoGPT
from config import LLM, EMBEDDINGS
from utils.performance import (
    track_performance,
    PerformanceStats,
    PerformanceCallbackHandler,
)


def initialize_vectorstore():
    """Initialize the vector store with Azure OpenAI embeddings."""
    embedding_size = 1536
    import faiss

    index = faiss.IndexFlatL2(embedding_size)
    return FAISS(
        embedding_function=EMBEDDINGS,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )


def initialize_tools(
    openweather_api_key: str, performance_stats: Optional[PerformanceStats] = None
) -> list[Tool]:
    """Initialize the tools for the agent."""
    # Create todo chain
    todo_prompt = PromptTemplate.from_template(
        "You are a planner who is an expert at coming up with a todo list for a given objective. "
        "Come up with a todo list for this objective: {objective}"
    )
    # Create callback handler
    callbacks = (
        [PerformanceCallbackHandler(performance_stats)] if performance_stats else None
    )

    # Using the new recommended way with callbacks
    todo_chain = LLMChain(llm=LLM, prompt=todo_prompt, callbacks=callbacks)

    # Create weather tool with explicit API key
    weather_wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=openweather_api_key
    )
    open_weather = OpenWeatherMapQueryRun(api_wrapper=weather_wrapper)

    # Wrap tool functions to track API calls
    def track_tool_usage(func):
        def wrapper(*args, **kwargs):
            if performance_stats:
                performance_stats.num_api_calls += 1
            return func(*args, **kwargs)

        return wrapper

    return [
        Tool(
            name="Weather",
            func=track_tool_usage(open_weather.run),
            description="useful for when you need to answer questions about the weather",
        ),
        Tool(
            name="TODO",
            func=track_tool_usage(todo_chain.run),
            description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. "
            "Output: a todo list for that objective. Please be very clear what the objective is!",
        ),
    ]


@track_performance
def main(performance_stats: Optional[PerformanceStats] = None):
    # Load environment variables
    load_dotenv()

    # Check for Azure OpenAI environment variables
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_VERSION",
        "DEPLOYMENT_NAME",
        "OPENWEATHER_API_KEY",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(
            f"Please set the following environment variables: {', '.join(missing_vars)}"
        )

    # Initialize components
    openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
    vectorstore = initialize_vectorstore()
    tools = initialize_tools(openweather_api_key, performance_stats)



    # Initialize AutoGPT
    agent = AutoGPT.from_llm_and_tools(
        ai_name="Tom",
        ai_role="Assistant",
        tools=tools,
        llm=LLM,
        memory=vectorstore.as_retriever(search_kwargs={"k": 8}),
        chat_history_memory=FileChatMessageHistory("chat_history.txt"),
        #callbacks=callbacks,
    )

    # Run example task
    agent.invoke(["write a weather report for San Francisco,US today and suggest a todo list"])


if __name__ == "__main__":
    main()
