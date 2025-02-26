"""BabyAGI agent."""

from collections import deque
from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, ConfigDict, Field

from .task_creation import (
    TaskCreationChain,
)
from .task_execution import (
    TaskExecutionChain,
)
from .task_prioritization import (
    TaskPrioritizationChain,
)


# This class has a metaclass conflict: both `Chain` and `BaseModel` define a metaclass
# to use, and the two metaclasses attempt to define the same functions but
# in mutually-incompatible ways. It isn't clear how to resolve this,
# and this code predates mypy beginning to perform that check.
#
# Mypy errors:
# ```
# Definition of "__repr_str__" in base class "Representation" is
#   incompatible with definition in base class "BaseModel"  [misc]
# Definition of "__repr_name__" in base class "Representation" is
#   incompatible with definition in base class "BaseModel"  [misc]
# Definition of "__rich_repr__" in base class "Representation" is
#   incompatible with definition in base class "BaseModel"  [misc]
# Definition of "__pretty__" in base class "Representation" is
#   incompatible with definition in base class "BaseModel"  [misc]
# Metaclass conflict: the metaclass of a derived class must be
#   a (non-strict) subclass of the metaclasses of all its bases  [misc]
# ```
#
# TODO: look into refactoring this class in a way that avoids the mypy type errors
class BabyAGI(Chain, BaseModel):  # type: ignore[misc]
    """Controller model for the BabyAGI agent."""

    task_list: deque = Field(default_factory=deque)
    task_creation_chain: Chain = Field(...)
    task_prioritization_chain: Chain = Field(...)
    execution_chain: Chain = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def add_task(self, task: Dict) -> None:
        self.task_list.append(task)

    def print_task_list(self) -> None:
        print(
            "\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m"
        )  # noqa: T201
        for t in self.task_list:
            task_name = t["task_name"] if isinstance(t, dict) else t
            task_id = t["task_id"] if isinstance(t, dict) else "1"
            print(f"{task_id}: {task_name}")  # noqa: T201  # noqa: T201

    def print_next_task(self, task: Dict) -> None:
        print(
            "\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m"
        )  # noqa: T201
        if isinstance(task, dict):
            task_name = task.get("task_name", "")
            task_id = task.get("task_id", "1")
            print(f"{task_id}: {task_name}")  # noqa: T201
        else:
            print(f"1: {task}")  # noqa: T201

    def print_task_result(self, result: str) -> None:
        print(
            "\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m"
        )  # noqa: T201
        print(result)  # noqa: T201

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def get_next_task(
        self, result: str, task_description: str, objective: str, **kwargs: Any
    ) -> List[Dict]:
        """Get the next task."""
        task_names = [t["task_name"] for t in self.task_list]
        incomplete_tasks = ", ".join(task_names)

        # Use invoke instead of run
        response = self.task_creation_chain.invoke(
            {
                "result": result,
                "task_description": task_description,
                "incomplete_tasks": incomplete_tasks,
                "objective": objective,
                **kwargs,
            }
        )

        # Extract the string result from the response
        response_text = (
            response.get("text", "") if isinstance(response, dict) else str(response)
        )
        new_tasks = response_text.split("\n")

        return [
            {"task_name": task_name} for task_name in new_tasks if task_name.strip()
        ]

    def prioritize_tasks(
        self, this_task_id: int, objective: str, **kwargs: Any
    ) -> List[Dict]:
        """Prioritize tasks."""
        task_names = [t["task_name"] for t in list(self.task_list)]
        next_task_id = int(this_task_id) + 1

        try:
            # Replace run with invoke
            response = self.task_prioritization_chain.invoke(
                {
                    "task_names": ", ".join(task_names),
                    "next_task_id": str(next_task_id),
                    "objective": objective,
                    **kwargs,
                }
            )

            # Extract the string result from the response
            response_text = (
                response.get("text", "")
                if isinstance(response, dict)
                else str(response)
            )
            new_tasks = response_text.split("\n")
            prioritized_task_list = []

            current_task_id = next_task_id
            for task_string in new_tasks:
                if not task_string.strip():
                    continue

                # Try to parse the task string in format "ID. Task Name"
                try:
                    task_parts = task_string.strip().split(".", 1)
                    if len(task_parts) == 2:
                        task_name = task_parts[1].strip()
                    else:
                        task_name = task_string.strip()

                    # Always use an incremental task ID
                    prioritized_task_list.append(
                        {"task_id": current_task_id, "task_name": task_name}
                    )
                    current_task_id += 1
                except Exception:
                    continue

            return (
                prioritized_task_list
                if prioritized_task_list
                else [
                    {
                        "task_id": next_task_id,
                        "task_name": "Review and analyze existing tasks",
                    }
                ]
            )

        except Exception as e:
            print(f"Error in task prioritization: {e}")  # noqa: T201
            # Return a safe default task if prioritization fails
            return [
                {
                    "task_id": next_task_id,
                    "task_name": "Review and analyze existing tasks",
                }
            ]

    def _get_top_tasks(self, query: str, k: int) -> List[str]:
        """Get the top k tasks based on the query."""
        results = self.vectorstore.similarity_search(query, k=k)
        if not results:
            return []
        return [str(item.metadata["task"]) for item in results]

    def execute_task(self, objective: str, task: str, k: int = 5, **kwargs: Any) -> str:
        """Execute a task."""
        # Get relevant context from previous tasks
        context = self._get_top_tasks(query=objective, k=k)

        # Remove callbacks from kwargs if it exists to prevent duplication
        kwargs.pop("callbacks", None)

        # Format the input dictionary with all required fields
        input_dict = {
            "input": task,
            "objective": objective,
            "task": task,
            "context": "\n".join(context),
            "tools": str(self.execution_chain.tools),
            "tool_names": [tool.name for tool in self.execution_chain.tools],
            "agent_scratchpad": kwargs.pop("agent_scratchpad", ""),
        }

        try:
            result = self.execution_chain.invoke(input_dict, **kwargs)
            return result.get("output", str(result))
        except Exception as e:
            print(f"Error executing task: {e}")  # noqa: T201
            return f"Task execution failed: {str(e)}"

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run the agent."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        objective = inputs["objective"]
        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0
        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = self.execute_task(
                    objective, task["task_name"], callbacks=_run_manager.get_child()
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)

                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}_{num_iters}"

                # Convert result to string if it's a list
                if isinstance(result, (tuple, list)):
                    result_text = str(result[0]) if result else ""
                else:
                    result_text = str(result)

                # Update the vectorstore add_texts call
                self.vectorstore.add_texts(
                    texts=[result_text],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = self.get_next_task(
                    result,
                    task["task_name"],
                    objective,
                    callbacks=_run_manager.get_child(),
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    self.prioritize_tasks(
                        this_task_id, objective, callbacks=_run_manager.get_child()
                    )
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print(  # noqa: T201
                    "\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m"
                )
                break
        # Return the result
        return {"result": result_text}

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        vectorstore: VectorStore,
        verbose: bool = False,
        task_execution_chain: Optional[Chain] = None,
        **kwargs: Any,
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(llm, verbose=verbose)
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        if task_execution_chain is None:
            execution_chain: Chain = TaskExecutionChain.from_llm(llm, verbose=verbose)
        else:
            execution_chain = task_execution_chain
        return cls(  # type: ignore[call-arg, call-arg, call-arg, call-arg]
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=execution_chain,
            vectorstore=vectorstore,
            **kwargs,
        )
