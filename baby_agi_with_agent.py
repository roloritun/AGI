"""BabyAGI with Tools Implementation

This script demonstrates how to implement BabyAGI with tools for real information gathering.
It builds on top of the base BabyAGI implementation by replacing the execution chain
with an agent that has access to tools.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain_community.tools import OpenWeatherMapQueryRun
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from pyowm.commons.exceptions import NotFoundError

from babyagi.baby_agi import BabyAGI
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


def safe_weather_run(weather_tool, location: str) -> str:
    """Safely run the weather tool with error handling."""
    try:
        # Expand common city abbreviations
        location_mapping = {
            "SF": "San Francisco,US",
            "NYC": "New York,US",
            "LA": "Los Angeles,US",
        }
        full_location = location_mapping.get(location.upper(), location)
        return weather_tool.run(full_location)
    except NotFoundError:
        return f"Could not find weather data for location: {location}. Please try with a more specific location name (e.g., 'San Francisco' instead of 'SF')."
    except Exception as e:
        return f"Error getting weather data: {str(e)}"


def create_tools(
    openweather_api_key: str, performance_stats: Optional[PerformanceStats] = None
) -> list[Tool]:
    """Create the tools for the agent to use."""
    # Create the todo chain
    todo_prompt = PromptTemplate.from_template(
        "You are a planner who is an expert at coming up with a todo list for a given objective. "
        "Come up with a todo list for this objective: {objective}"
    )
    callbacks = (
        [PerformanceCallbackHandler(performance_stats)] if performance_stats else None
    )
    todo_chain = LLMChain(llm=LLM, prompt=todo_prompt, callbacks=callbacks)

    # Create the weather tool
    weather_wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=openweather_api_key
    )
    open_weather = OpenWeatherMapQueryRun(api_wrapper=weather_wrapper)

    # Wrap tool functions to track API calls if stats are provided
    def track_tool_usage(func):
        def wrapper(*args, **kwargs):
            if performance_stats:
                performance_stats.num_api_calls += 1
            return func(*args, **kwargs)

        return wrapper

    return [
        Tool(
            name="Weather",
            func=track_tool_usage(lambda x: safe_weather_run(open_weather, x)),
            description="useful for when you need to answer questions about the weather. Input should be a city name (e.g., 'San Francisco' not 'SF').",
        ),
        Tool(
            name="TODO",
            func=track_tool_usage(todo_chain.run),
            description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. "
            "Output: a todo list for that objective. Please be very clear what the objective is!",
        ),
    ]


def create_agent(
    tools: list[Tool], performance_stats: Optional[PerformanceStats] = None
) -> AgentExecutor:
    """Create the agent executor with the given tools and LLM."""
    # Create the agent prompt
    prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
    suffix = """Question: {task}
{agent_scratchpad}"""
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["objective", "task", "context", "agent_scratchpad"],
    )

    # Create the LLM chain with callback handler
    callbacks = (
        [PerformanceCallbackHandler(performance_stats)] if performance_stats else None
    )
    llm_chain = LLMChain(llm=LLM, prompt=prompt, callbacks=callbacks)

    # Create the agent
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)

    # Create and return the agent executor
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,  # Prevent infinite loops
        handle_parsing_errors=True,
        callbacks=callbacks,  # Add callbacks to the executor as well
    )


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
    tools = create_tools(openweather_api_key, performance_stats)
    agent_executor = create_agent(tools, performance_stats)

    # Configure BabyAGI
    OBJECTIVE = "Write a weather report for San Francisco,US today"
    max_iterations: Optional[int] = 3

    # Create callback handler for BabyAGI
    callbacks = (
        [PerformanceCallbackHandler(performance_stats)] if performance_stats else None
    )

    # Create and run BabyAGI
    baby_agi = BabyAGI.from_llm(
        llm=LLM,
        vectorstore=vectorstore,
        task_execution_chain=agent_executor,
        verbose=True,
        max_iterations=max_iterations,
        callbacks=callbacks,  # Add callbacks to BabyAGI
    )

    baby_agi.invoke({"objective": OBJECTIVE})


if __name__ == "__main__":
    main()
