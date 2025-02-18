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
            f"Performance Statistics:\n"
            f"Duration: {self.calculate_duration():.2f} seconds\n"
            f"Total Tokens: {self.total_tokens:,}\n"
            f"- Prompt Tokens: {self.prompt_tokens:,}\n"
            f"- Completion Tokens: {self.completion_tokens:,}\n"
            f"Total Cost: ${self.total_cost:.4f}\n"
            f"Number of API Calls: {self.num_api_calls}\n"
            f"{'='*50}\n"
        )


class PerformanceCallbackHandler(BaseCallbackHandler):
    """Callback handler for tracking performance metrics."""

    def __init__(self, stats: PerformanceStats):
        """Initialize with performance stats object."""
        self.stats = stats

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Track token usage when LLM call ends."""
        if not self.stats:
            return

        # Update token counts from usage info
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.stats.prompt_tokens += usage.get("prompt_tokens", 0)
            self.stats.completion_tokens += usage.get("completion_tokens", 0)
            self.stats.total_tokens += usage.get("total_tokens", 0)

            # Estimate cost (assuming GPT-4 pricing)
            prompt_cost = (
                usage.get("prompt_tokens", 0) * 0.03
            ) / 1000  # $0.03 per 1k tokens
            completion_cost = (
                usage.get("completion_tokens", 0) * 0.06
            ) / 1000  # $0.06 per 1k tokens
            self.stats.total_cost += prompt_cost + completion_cost

        self.stats.num_api_calls += 1


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
