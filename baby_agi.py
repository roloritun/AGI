"""BabyAGI Implementation

This script demonstrates how to implement BabyAGI, an AI agent that can generate
and pretend to execute tasks based on a given objective.
"""

import os
from typing import Optional
from dotenv import load_dotenv

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from babyagi.baby_agi import BabyAGI
from config import LLM, EMBEDDINGS


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


def main():
    # Load environment variables
    load_dotenv()

    # Check for Azure OpenAI environment variables
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_VERSION",
        "DEPLOYMENT_NAME",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(
            f"Please set the following environment variables: {', '.join(missing_vars)}"
        )

    # Initialize components
    vectorstore = initialize_vectorstore()

    # Configure BabyAGI
    OBJECTIVE = "Write a weather report for SF today"
    max_iterations: Optional[int] = 3

    # Create and run BabyAGI
    baby_agi = BabyAGI.from_llm(
        llm=LLM,
        vectorstore=vectorstore,
        verbose=True,
        max_iterations=max_iterations,
    )

    baby_agi.invoke({"objective": OBJECTIVE})


if __name__ == "__main__":
    main()
