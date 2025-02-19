"""Performance tracking utilities for AI agents."""

import time
from typing import Any, Callable, Dict, Optional, List
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


@dataclass
class PerformanceStats:
    """Container for performance statistics."""

    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    num_api_calls: int = 0
    num_llm_calls: int = 0
    num_chain_calls: int = 0
    num_tool_calls: int = 0
    tool_usage: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def calculate_duration(self) -> float:
        """Calculate duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "duration_seconds": self.calculate_duration(),
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_cost": self.total_cost,
            "num_api_calls": self.num_api_calls,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

    def __str__(self) -> str:
        """Format stats as string."""
        return (
            f"\n{'='*50}\n"
            #f"Performance Statistics:\n"
            f"Duration: {self.calculate_duration():.2f} seconds\n"
            #f"API Calls Breakdown:\n"
            # f"- Total API Calls: {self.num_api_calls}\n"
            # f"- LLM Calls: {self.num_llm_calls}\n"
            # f"- Chain Calls: {self.num_chain_calls}\n"
            # f"- Tool Calls: {self.num_tool_calls}\n"
            # f"Token Usage:\n"
            # f"- Total Tokens: {self.total_tokens:,}\n"
            # f"- Prompt Tokens: {self.prompt_tokens:,}\n"
            # f"- Completion Tokens: {self.completion_tokens:,}\n"
            # f"Total Cost: ${self.total_cost:.4f}\n"
            f"{'='*50}\n"
        )

    def log_tool_usage(self, tool_name: str) -> None:
        """Track tool usage."""
        self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1

    def generate_report(self) -> str:
        """Generate a formatted performance report."""
        duration = (self.end_time or datetime.now()) - self.start_time
        report = [
           # "Performance Report",
            "==================",
            f"Duration: {duration.total_seconds():.2f} seconds",
           # f"Total API Calls: {self.num_api_calls}",
           # f"Total Tokens: {self.total_tokens}",
          # f" - Prompt Tokens: {self.prompt_tokens}",
           # f" - Completion Tokens: {self.completion_tokens}",
           # f"Total Cost: ${self.total_cost:.4f}",
           # "\nTool Usage:",
        ]

        # for tool, count in self.tool_usage.items():
        #     report.append(f" - {tool}: {count} calls")

        # if self.errors:
        #     report.extend(["\nErrors:", *[f" - {err}" for err in self.errors]])

        return "\n".join(report)


class PerformanceCallbackHandler(BaseCallbackHandler):
    """Callback handler for tracking performance metrics."""

    def __init__(self, stats: PerformanceStats):
        """Initialize with performance stats object."""
        super().__init__()
        self.stats = stats

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Log when LLM starts."""
        if self.stats:
            self.stats.num_api_calls += 1
            self.stats.num_llm_calls += 1

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Log when chain starts."""
        if self.stats:
            self.stats.num_api_calls += 1
            self.stats.num_chain_calls += 1

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Log when a tool starts."""
        if self.stats:
            tool_name = serialized.get("name", "unknown_tool")
            self.stats.log_tool_usage(tool_name)
            # Count tool calls as API calls
            self.stats.num_api_calls += 1
            self.stats.num_tool_calls += 1

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Log tool errors."""
        if self.stats:
            self.stats.errors.append(f"Tool error: {str(error)}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Track token usage and costs when LLM call ends."""
        if not self.stats or not hasattr(response, "llm_output"):
            return

        llm_output = response.llm_output
        if not isinstance(llm_output, dict) or "token_usage" not in llm_output:
            return

        # Extract token usage
        token_usage = llm_output["token_usage"]
        completion_tokens = token_usage.get("completion_tokens", 0)
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)

        # Update token counts
        self.stats.completion_tokens += completion_tokens
        self.stats.prompt_tokens += prompt_tokens
        self.stats.total_tokens += total_tokens

        # Get model name and standardize it
        model_name = llm_output.get("model_name", "gpt-4").lower()

        # Define cost rates based on model
        cost_rates = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
            "text-davinci-003": {"prompt": 0.02, "completion": 0.02},
        }

        # Get rates for the model, default to GPT-4 rates
        model_rates = cost_rates.get(model_name, cost_rates["gpt-4"])

        # Calculate costs
        prompt_cost = (prompt_tokens * model_rates["prompt"]) / 1000
        completion_cost = (completion_tokens * model_rates["completion"]) / 1000
        total_cost = prompt_cost + completion_cost

        # Update total cost
        self.stats.total_cost += total_cost

        # Log detailed metrics if verbose


def track_performance(func: Callable) -> Callable:
    """Decorator to track performance metrics of AI agent executions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Initialize performance tracking
        stats = PerformanceStats()

        # Add stats to kwargs for the function to update
        kwargs["performance_stats"] = stats

        try:
            # Execute the function
            result = func(*args, **kwargs)
            return result
        finally:
            # Record end time
            stats.end_time = datetime.now()
            # Print performance statistics
            print(stats)

    return wrapper
