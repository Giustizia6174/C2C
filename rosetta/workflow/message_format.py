from abc import ABC, abstractmethod

# =============================================================================
# Model-specific content formatters
# =============================================================================

class ContentFormatter(ABC):
    """Base class for model-specific content formatting."""

    @abstractmethod
    def format_assistant(self, content: str, reasoning: str = None) -> str:
        """Format assistant message with optional reasoning."""
        pass

    @abstractmethod
    def format_tool_result(self, tool_name: str, result: str) -> str:
        """Format tool result."""
        pass


class DefaultFormatter(ContentFormatter):
    """Default formatter using XML-style tags."""

    def format_assistant(self, content: str, reasoning: str = None) -> str:
        parts = []
        if reasoning:
            parts.append(f"<think>{reasoning}</think>")
        if content:
            parts.append(content)
        return "\n".join(parts)

    def format_tool_result(self, tool_name: str, result: str) -> str:
        return result


class QwenFormatter(ContentFormatter):
    """Formatter for Qwen models using <think> tags."""

    def format_assistant(self, content: str, reasoning: str = None) -> str:
        parts = []
        if reasoning:
            parts.append(f"<think>\n{reasoning}\n</think>")
        if content:
            parts.append(content)
        return "\n".join(parts)

    def format_tool_result(self, tool_name: str, result: str) -> str:
        return result


class GPTOSSFormatter(ContentFormatter):
    """Formatter for GPT-OSS models using channel tags."""

    def format_assistant(self, content: str, reasoning: str = None) -> str:
        parts = []
        if reasoning:
            parts.append(f"<|channel|>analysis<|message|>{reasoning}<|end|>")
        if content:
            parts.append(f"<|channel|>final<|message|>{content}")
        return "".join(parts)

    def format_tool_result(self, tool_name: str, result: str) -> str:
        return result


# Formatter registry
FORMATTERS = {
    "default": DefaultFormatter,
    "qwen": QwenFormatter,
    "gpt-oss": GPTOSSFormatter,
}

# Keyword patterns for auto-detection (checked in order)
MODEL_PATTERNS = [
    (["gpt-oss", "gptoss"], "gpt-oss"),
    (["qwen", "qwen2", "qwen3"], "qwen"),
    (["llama", "llama2", "llama3"], "default"),
    (["mistral", "mixtral"], "default"),
    (["deepseek"], "default"),
]


def detect_model_type(model_name: str) -> str:
    """Detect formatter type from model name using keyword matching.

    Args:
        model_name: Model name like "openai/gpt-oss-120b" or "Qwen/Qwen3-0.6B"

    Returns:
        Formatter type string ("qwen", "gpt-oss", "default")
    """
    name_lower = model_name.lower()
    for keywords, formatter_type in MODEL_PATTERNS:
        if any(kw in name_lower for kw in keywords):
            return formatter_type
    return "default"


def get_formatter(model_name_or_type: str = "default") -> ContentFormatter:
    """Get formatter by model type or model name.

    Args:
        model_name_or_type: Either a formatter type ("qwen", "gpt-oss", "default")
            or a model name ("openai/gpt-oss-120b", "Qwen/Qwen3-0.6B")

    Returns:
        ContentFormatter instance
    """
    # Check if it's a direct formatter type
    if model_name_or_type in FORMATTERS:
        return FORMATTERS[model_name_or_type]()

    # Otherwise, try to detect from model name
    formatter_type = detect_model_type(model_name_or_type)
    return FORMATTERS[formatter_type]()