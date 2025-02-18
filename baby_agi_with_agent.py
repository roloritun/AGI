"""BabyAGI with Tools Implementation

This script demonstrates how to implement BabyAGI with tools for real information gathering.
It builds on top of the base BabyAGI implementation by replacing the execution chain
with an agent that has access to tools.
"""

import os
from typing import Optional
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain_community.tools import OpenWeatherMapQueryRun
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper

from babyagi.baby_agi import BabyAGI


def initialize_vectorstore():
    """Initialize the vector store with OpenAI embeddings."""
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    import faiss

    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embedding_function=embeddings_model,  # Use the embeddings model directly
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )
    return vectorstore


def create_tools(llm: OpenAI, openweather_api_key: str) -> list[Tool]:
    """Create the tools for the agent to use."""
    # Create the todo chain
    todo_prompt = PromptTemplate.from_template(
        "You are a planner who is an expert at coming up with a todo list for a given objective. "
        "Come up with a todo list for this objective: {objective}"
    )
    todo_chain = LLMChain(llm=llm, prompt=todo_prompt)

    # Create the weather tool
    weather_wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=openweather_api_key
    )
    open_weather = OpenWeatherMapQueryRun(api_wrapper=weather_wrapper)

    return [
        Tool(
            name="Weather",
            func=open_weather.run,
            description="useful for when you need to answer questions about the weather",
        ),
        Tool(
            name="TODO",
            func=todo_chain.run,
            description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. "
            "Output: a todo list for that objective. Please be very clear what the objective is!",
        ),
    ]


def create_agent(tools: list[Tool], llm: ChatOpenAI) -> AgentExecutor:
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

    # Create the LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

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
    )


def main():
    # Load environment variables
    load_dotenv()

    # Check for API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openweather_api_key = os.getenv("OPENWEATHER_API_KEY")

    if not openai_api_key or not openweather_api_key:
        raise ValueError(
            "Please set OPENAI_API_KEY and OPENWEATHER_API_KEY environment variables"
        )

    # Initialize models and tools
    llm = OpenAI(temperature=0)
    chat_llm = ChatOpenAI(temperature=0)
    vectorstore = initialize_vectorstore()
    tools = create_tools(llm, openweather_api_key)
    agent_executor = create_agent(tools, chat_llm)

    # Configure BabyAGI
    OBJECTIVE = "Write a weather report for SF today"
    max_iterations: Optional[int] = 3

    # Create and run BabyAGI
    baby_agi = BabyAGI.from_llm(
        llm=chat_llm,
        vectorstore=vectorstore,
        tools=tools,  # Pass the tools list
        verbose=True,
        max_iterations=max_iterations,
    )

    baby_agi.invoke({"objective": OBJECTIVE})


if __name__ == "__main__":
    main()
