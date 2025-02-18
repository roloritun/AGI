from langchain.chains import LLMChain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate


class TaskExecutionChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        execution_template = (
            "You are an AI who performs one task based on the following objective: {objective}\n"
            "Take into account these previously completed tasks: {context}\n\n"
            "Your task: {task}\n\n"
            "Follow these guidelines:\n"
            "1. If the task involves searching or finding information, be specific about sources\n"
            "2. If the task involves analysis, show your reasoning step by step\n"
            "3. If you encounter any limitations, explain them clearly\n"
            "4. If you need to make assumptions, state them explicitly\n\n"
            "Response:"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

    def _process_response(self, response: str) -> str:
        """Process the response to ensure it's useful."""
        if not response.strip():
            return "Task could not be completed. Please try breaking it into smaller subtasks."

        # Add error handling if needed
        if "error" in response.lower() or "unable to" in response.lower():
            return f"Task encountered difficulties: {response}\nSuggestion: Break this task into smaller, more specific subtasks."

        return response
