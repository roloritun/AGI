from langchain.chains import LLMChain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate


class TaskCreationChain(LLMChain):
    """Chain generating tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as a numbered list, with each task on a new line,"
            " prefixed by the task number and a period, like:"
            " 1. First task description"
            " 2. Second task description"
            " Make tasks specific, actionable, and directed toward the objective."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=[
                "result",
                "task_description",
                "incomplete_tasks",
                "objective",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

    def _process_output(self, output: str) -> str:
        """Process the output to ensure it's properly formatted."""
        # Clean up the output and ensure proper formatting
        lines = output.strip().split("\n")
        processed_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line starts with a number, if not add one
            if not any(line.startswith(f"{i}.") for i in range(1, 10)):
                processed_lines.append(f"1. {line}")
            else:
                processed_lines.append(line)

        return (
            "\n".join(processed_lines)
            if processed_lines
            else "1. Review previous tasks and create new subtasks"
        )
