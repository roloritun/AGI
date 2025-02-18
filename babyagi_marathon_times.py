"""Marathon Times Analysis

This script uses BabyAGI to find and analyze winning marathon times.
Uses LangChain primitives (LLMs, PromptTemplates, VectorStores, Embeddings, Tools).
"""

import os
import asyncio
from contextlib import contextmanager
from typing import Optional
from dotenv import load_dotenv

import nest_asyncio
import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from langchain.agents import tool
from langchain.docstore.document import Document
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)
from langchain.agents import Tool
from langchain_community.tools.file_management.read import ReadFileTool
from tools.file_tools import CustomWriteFileTool
from langchain.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import ToolException
from langchain.chains.qa_with_sources.loading import (
    BaseCombineDocumentsChain,
    load_qa_with_sources_chain,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from pydantic import Field
from langchain_community.tools.human.tool import HumanInputRun

from babyagi.baby_agi import BabyAGI

from config import LLM, EMBEDDINGS

from utils.performance import (
    track_performance,
    PerformanceStats,
    PerformanceCallbackHandler,
)

from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

# Needed since jupyter runs an async eventloop
nest_asyncio.apply()

ROOT_DIR = "./data2/"

# Create data directory if it doesn't exist
os.makedirs(ROOT_DIR, exist_ok=True)


def get_input() -> str:
    print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)


@contextmanager
def pushd(new_dir):
    """Context manager for changing the current working directory."""
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)


@tool
@track_performance
def process_csv(
    csv_file_path: str, instructions: str, output_path: Optional[str] = None
) -> str:
    """Process a CSV by with pandas in a limited REPL.
    Only use this after writing data to disk as a csv file.
    Any figures must be saved to disk to be viewed by the human.
    Instructions should be written in natural language, not code."""
    with pushd(ROOT_DIR):
        try:
            df = pd.read_csv(csv_file_path)
        except Exception as e:
            return f"Error: {e}"
        agent = create_pandas_dataframe_agent(LLM, df, max_iterations=30, verbose=True)
        if output_path is not None:
            instructions += f" Save output to disk at {output_path}"
        try:
            result = agent.run(instructions)
            return result
        except Exception as e:
            return f"Error: {e}"


async def async_load_playwright(url: str) -> str:
    """Load the specified URLs using Playwright and parse using BeautifulSoup."""
    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            results = "\n".join(chunk for chunk in chunks if chunk)
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    return results


def run_async(coro):
    event_loop = asyncio.get_event_loop()
    return event_loop.run_until_complete(coro)


@tool
def browse_web_page(url: str) -> str:
    """Verbose way to scrape a whole webpage."""
    return run_async(async_load_playwright(url))


def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )


class WebpageQATool(BaseTool):
    name: str = "query_webpage"
    description: str = (
        "Browse a webpage and answer questions about it. Input should be formatted as 'url||question'"
    )
    text_splitter: RecursiveCharacterTextSplitter = Field(
        default_factory=_get_text_splitter
    )
    qa_chain: BaseCombineDocumentsChain

    def _run(self, query: str) -> str:
        try:
            url, question = query.split("||")
        except ValueError:
            raise ToolException("Please provide input in the format: url||question")

        result = browse_web_page.run(url)
        docs = [Document(page_content=result, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)

        results = []
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i : i + 4]
            window_result = self.qa_chain.invoke(
                {"input_documents": input_docs, "question": question}
            )
            results.append(f"Response from window {i} - {window_result}")

        results_docs = [
            Document(page_content="\n".join(results), metadata={"source": url})
        ]
        return self.qa_chain.invoke(
            {"input_documents": results_docs, "question": question}
        )

    async def _arun(self, query: str) -> str:
        raise NotImplementedError


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


def initialize_tools(llm: LLM) -> list[Tool]:
    """Initialize all tools for the agent."""
    web_search = DuckDuckGoSearchRun()
    query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))

    return [
        web_search,
        Tool(
            name="write_file",
            func=lambda x: CustomWriteFileTool().invoke(
                {"file_path": "./data2/" + x.split("||")[0], "text": x.split("||")[1]}
            ),
            description="Write a file to disk. Input should be formatted as 'file_path||content'",
        ),
        ReadFileTool(root_dir="./data2"),
        process_csv,
        query_website_tool,
        HumanInputRun(input_func=get_input),
    ]


def create_agent(tools, performance_stats):
    # Update the prompt template with output key
    prompt = PromptTemplate(
        input_variables=[
            "objective",
            "task",
            "context",
            "tools",
            "tool_names",
            "agent_scratchpad",
        ],
        template="""
            You are working on the following objective: {objective}
            Current task: {task}
            Previous context: {context}

            You have access to the following tools:
            {tools}

            Use the following format:
            Question: the task you must complete
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know what to do
            Final Answer: the final result of executing the task

            Begin!
            Question: {task}
            Thought: {agent_scratchpad}""",
    )
    # Create the agent with specific output key
    agent = create_react_agent(llm=LLM, tools=tools, prompt=prompt)

    # Create the agent executor with output key
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="force",
        return_intermediate_steps=True,
        output_key="output",  # Add this line to specify the output key
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
    vectorstore = initialize_vectorstore()
    tools = initialize_tools(LLM)
    agent_executor = create_agent(tools, performance_stats)

    # Configure BabyAGI with more specific first task
    OBJECTIVE = """What were the winning boston marathon times for the past 5 years (ending in 2022)? 
             Generate a table of the year, name, country of origin, and write as csv."""
    FIRST_TASK = "Search for Boston Marathon winners and their times from 2018-2022"
    max_iterations: Optional[int] = 3

    # Define callbacks
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
        callbacks=callbacks,
    )

    # Update the invoke call with simplified input structure
    result = baby_agi.invoke(
        {
            "objective": OBJECTIVE,
            "first_task": FIRST_TASK,  # Simplify to just pass the string
        }
    )

    return result


if __name__ == "__main__":
    main()
