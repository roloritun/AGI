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
from langchain_community.tools.file_management.write import WriteFileTool
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.chains.qa_with_sources.loading import (
    BaseCombineDocumentsChain,
    load_qa_with_sources_chain,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from pydantic import Field

from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from babyagi.baby_agi import BabyAGI
from langchain.chains import LLMChain

from config import LLM, EMBEDDINGS

from utils.performance import (
    track_performance,
    PerformanceStats,
    PerformanceCallbackHandler,
)

# Needed since jupyter runs an async eventloop
nest_asyncio.apply()

ROOT_DIR = "./data2/"

# Create data directory if it doesn't exist
os.makedirs(ROOT_DIR, exist_ok=True)


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
        """Useful for browsing websites and answering questions about them.
        Input should be formatted as 'url||question'"""
        try:
            url, question = query.split("||")
        except ValueError:
            return "Please provide input in the format: url||question"

        result = browse_web_page.run(url)
        docs = [Document(page_content=result, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i : i + 4]
            window_result = self.qa_chain(
                {"input_documents": input_docs, "question": question},
                return_only_outputs=True,
            )
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [
            Document(page_content="\n".join(results), metadata={"source": url})
        ]
        return self.qa_chain(
            {"input_documents": results_docs, "question": question},
            return_only_outputs=True,
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


def safe_write_file(input_str: str, text: str = None) -> str:
    """Safely write text to a file with proper error handling.
    Can be called in two ways:
    1. safe_write_file(file_path, text)
    2. safe_write_file("file_path||text")
    """
    try:
        if text is None:
            # Assume input is in format "file_path||text"
            if "||" not in input_str:
                return "Error: When providing single argument, use format 'file_path||text'"
            file_path, text = input_str.split("||", 1)
        else:
            # Using two-argument format
            file_path = input_str
            
        # Clean up file path
        file_path = file_path.strip()
        
        # Generate filename from content if none provided
        if not file_path:
            words = text.split()[:3]  # Take first 3 words
            file_path = "_".join(word.lower() for word in words if word.isalnum())
            if not file_path:
                file_path = "output"
        
        # Add .txt extension if missing
        if not file_path.endswith('.txt'):
            file_path += '.txt'
            
        # Ensure filename is unique
        base_path = os.path.splitext(file_path)[0]
        ext = os.path.splitext(file_path)[1]
        counter = 1
        while os.path.exists(os.path.join(ROOT_DIR, file_path)):
            file_path = f"{base_path}_{counter}{ext}"
            counter += 1
            
        # Write the file
        with open(os.path.join(ROOT_DIR, file_path), 'w') as f:
            f.write(text)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"


def initialize_tools(llm: LLM) -> list[Tool]:
    """Initialize all tools for the agent."""
    web_search = DuckDuckGoSearchRun()
    query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))

    # Create a wrapped version of WriteFileTool
    write_tool = Tool(
        name="write_file",
        func=safe_write_file,
        description="""Write text to a file. Can be used in two ways:
        1. Single argument: 'filename||content' (will generate filename if empty)
        2. Two arguments: filename and content
        Will add .txt extension automatically and ensure unique filename."""
    )

    return [
        web_search,
        write_tool,  # Use our wrapped version instead
        ReadFileTool(root_dir=ROOT_DIR),
        process_csv,
        query_website_tool,
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
    vectorstore = initialize_vectorstore()
    tools = initialize_tools(LLM)
    agent_executor = create_agent(tools, performance_stats)

    # Configure BabyAGI with more specific first task
    OBJECTIVE = """What were the winning boston marathon times for the past 5 years (ending in 2022)? 
             Generate a table of the year, name, country of origin, and times and write to marathon_results.txt."""
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
        callbacks=callbacks,  # Add callbacks to BabyAGI
    )

    baby_agi.invoke({"objective": OBJECTIVE, "first_task": FIRST_TASK})


if __name__ == "__main__":
    main()
