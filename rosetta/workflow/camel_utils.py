"""Utility functions for CAMEL message conversion."""

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Any
import json
from dotenv import load_dotenv, find_dotenv

from camel.agents import ChatAgent
from camel.messages import BaseMessage, FunctionCallingMessage
from camel.memories import MemoryRecord, ContextRecord
from camel.toolkits import FunctionTool
from camel.types import OpenAIBackendRole, RoleType, ModelPlatformType, ModelType, ChatCompletion
from camel.models import ModelFactory, BaseModelBackend
from camel.configs import ChatGPTConfig
from openai import Stream
from openai.types.chat import ChatCompletionChunk

def setup_env():
    """Setup environment variables."""
    load_dotenv(find_dotenv())

def create_model(
    provider: str,
    model_type: Optional[str] = None,
    model_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 32768,
    stream: Optional[bool] = None,
    chat_template_kwargs: Optional[dict] = None,
    **kwargs: Any,
):
    """Create a model based on the provider.

    Args:
        provider: Model provider, one of "local", "openai", "gemini".
        model_type: Model type/name string. If None, uses provider defaults:
            - local: "local"
            - openai: GPT_4O_MINI
            - gemini: "gemini-3-flash-preview"
        model_url: API URL for local/compatible models.
        api_key: API key (uses env var if not provided).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        chat_template_kwargs: Custom chat template kwargs for local models.
            Defaults to {"enable_thinking": False} if not provided.
        **kwargs: Additional model config parameters.

    Returns:
        Configured CAMEL model instance.

    Raises:
        ValueError: If provider is not supported.
    """
    if provider == "local":
        model_type = model_type or "local"
        # Use provided chat_template_kwargs or default
        effective_chat_template_kwargs = chat_template_kwargs if chat_template_kwargs is not None else {"enable_thinking": False}
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            model_config_dict={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_body": {"chat_template_kwargs": effective_chat_template_kwargs},
                **kwargs,
            },
            api_key=api_key or "not-needed",
            url=model_url or "http://localhost:30000/v1",
        )
    elif provider == "openai":
        model_type = model_type or ModelType.GPT_5_MINI
        config = ChatGPTConfig(max_tokens=max_tokens, temperature=temperature)
        model_config_dict = config.as_dict()
        if stream:
            model_config_dict["stream"] = True
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=model_type,
            model_config_dict=model_config_dict,
            api_key=api_key,
        )
    elif provider == "gemini":
        model_type = model_type or "gemini-3-flash-preview"
        config = ChatGPTConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=kwargs.get("reasoning_effort", "medium"),
        )
        model_config_dict = config.as_dict()
        if stream:
            model_config_dict["stream"] = True
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            model_config_dict=model_config_dict,
            api_key=api_key or os.getenv("GEMINI_API_KEY"),
            url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    elif provider == "fireworks":
        model_type = model_type or "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507"
        config = ChatGPTConfig(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        model_config_dict = config.as_dict()
        if stream:
            model_config_dict["stream"] = True
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            model_config_dict=model_config_dict,
            api_key=api_key or os.getenv("FIREWORKS_API_KEY"),
            url="https://api.fireworks.ai/inference/v1",
        )
    else:
        raise ValueError(
            f"Unsupported model provider: {provider}. "
            f"Choose from: local, openai, gemini"
        )


def collect_stream_response(
    stream: Stream[ChatCompletionChunk],
) -> ChatCompletion:
    """Consume a streaming response and return a complete ChatCompletion.

    This function iterates through all chunks in a streaming response,
    accumulates the content, reasoning_content, and tool calls, and constructs
    a complete ChatCompletion object that matches the non-streaming API format.

    Args:
        stream: A Stream of ChatCompletionChunk objects from the model.

    Returns:
        ChatCompletion: A complete response object with all accumulated content,
            including reasoning_content if the model provides it.
    """
    collected_content = ""
    collected_reasoning = ""
    collected_tool_calls = {}  # index -> {id, type, function: {name, arguments}}
    finish_reason = None
    model = None
    completion_id = None
    created = None
    usage = None

    for chunk in stream:
        # Capture metadata from first chunk
        if completion_id is None:
            completion_id = chunk.id
            created = chunk.created
            model = chunk.model

        # Check for usage in chunk (some providers send it in final chunk)
        if hasattr(chunk, 'usage') and chunk.usage is not None:
            usage = chunk.usage

        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta = choice.delta

        # Accumulate content
        if delta.content:
            collected_content += delta.content

        # Accumulate reasoning content (various field names used by different providers)
        # DeepSeek, Qwen3 use reasoning_content; some use thinking_content or reasoning
        reasoning_delta = None
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            reasoning_delta = delta.reasoning_content
        elif hasattr(delta, 'thinking_content') and delta.thinking_content:
            reasoning_delta = delta.thinking_content
        elif hasattr(delta, 'reasoning') and delta.reasoning:
            reasoning_delta = delta.reasoning

        if reasoning_delta:
            collected_reasoning += reasoning_delta

        # Accumulate tool calls
        if delta.tool_calls:
            for tc in delta.tool_calls:
                tc_index = tc.index

                if tc_index not in collected_tool_calls:
                    collected_tool_calls[tc_index] = {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }

                # Update id if we get it (usually only in first chunk for this tool call)
                if tc.id:
                    collected_tool_calls[tc_index]["id"] = tc.id

                # Accumulate function name and arguments
                if tc.function:
                    if tc.function.name:
                        collected_tool_calls[tc_index]["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                        collected_tool_calls[tc_index]["function"]["arguments"] += tc.function.arguments

        # Capture finish reason
        if choice.finish_reason:
            finish_reason = choice.finish_reason

    # Build tool_calls list if any were collected
    tool_calls = None
    if collected_tool_calls:
        # Sort by index and filter out any with missing id
        tool_calls = [
            collected_tool_calls[i]
            for i in sorted(collected_tool_calls.keys())
            if collected_tool_calls[i]["id"]
        ]

    # Construct the ChatCompletion object
    from openai.types.chat import ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    message = ChatCompletionMessage(
        role="assistant",
        content=collected_content or None,
        tool_calls=tool_calls if tool_calls else None,
    )

    # Add reasoning_content as attribute if present
    if collected_reasoning:
        message.reasoning_content = collected_reasoning

    choice_obj = Choice(
        index=0,
        message=message,
        finish_reason=finish_reason or "stop",
    )

    # Use collected usage from stream or create placeholder
    if usage is None:
        usage = CompletionUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    return ChatCompletion(
        id=completion_id or "stream_collected",
        choices=[choice_obj],
        created=created or 0,
        model=model or "unknown",
        object="chat.completion",
        usage=usage,
    )


def model_run_sync(
    model: BaseModelBackend,
    messages: List[dict],
    tools: Optional[List[dict]] = None,
) -> ChatCompletion:
    """Run a model and return a complete ChatCompletion, handling streaming transparently.

    This wrapper function calls model.run() and handles both streaming and
    non-streaming responses. If the model is configured for streaming, it
    consumes the entire stream and returns a complete ChatCompletion object.

    Args:
        model: A CAMEL BaseModelBackend instance.
        messages: List of message dicts in OpenAI format.
        tools: Optional list of tool schemas.

    Returns:
        ChatCompletion: A complete response object, regardless of streaming mode.
    """
    response = model.run(messages, tools=tools)

    # Check if response is a stream (handles both openai.Stream and _SyncStreamWrapper)
    if isinstance(response, Stream):
        return collect_stream_response(response)

    # Check for other streaming wrapper types (e.g., _SyncStreamWrapper)
    # These are iterable and don't have .choices attribute
    if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
        return collect_stream_response(response)

    # Non-streaming response - return as-is
    return response


def context_records_to_memory_records(
    records: List[ContextRecord]
) -> List[MemoryRecord]:
    """Convert ContextRecord list to standard message format.
    
    Args:
        records: List of MemoryRecord objects.
    """
    return [ctx_record.memory_record for ctx_record in records]

def memoryRecord_flip_role(message: MemoryRecord) -> MemoryRecord:
    """Flip the role of a message."""
    if message.message.role_type == RoleType.USER:
        message.message.role_type = RoleType.ASSISTANT
    elif message.message.role_type == RoleType.ASSISTANT:
        message.message.role_type = RoleType.USER
    elif message.message.role_type == RoleType.SYSTEM:
        message.message.role_type = RoleType.SYSTEM
    elif message.message.role_type == RoleType.FUNCTION:
        message.message.role_type = RoleType.FUNCTION
    elif message.message.role_type == RoleType.TOOL:
        message.message.role_type = RoleType.TOOL
    else:
        raise ValueError(f"Unsupported role type: {message.message.role_type}.")
    return message

def messages_to_memoryRecords(
    chat_history: List[dict],
    skip_system: bool = False
) -> List[MemoryRecord]:
    """Convert standard message format to CAMEL MemoryRecord list.

    Args:
        chat_history: List of dictionaries with 'role' and 'content' keys.
                     Roles can be 'user', 'assistant', 'system', 'function',
                     'tool', or 'developer'.
        skip_system: Whether to skip system messages. Default is True.

    Returns:
        List of MemoryRecord objects suitable for CAMEL agents.

    Example:
        >>> chat_history = [
        ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
        ...     {'role': 'user', 'content': 'Hello'},
        ...     {'role': 'assistant', 'content': 'Hi there!'}
        ... ]
        >>> message_list = convert_to_camel_messages(chat_history)
        >>> len(message_list)  # System message skipped by default
        2
    """
    message_list = []

    # Build a mapping of tool_call_id -> function_name for tool messages
    # that don't have func_name specified
    tool_call_map = {}
    for msg in chat_history:
        if msg.get('role') == 'assistant' and 'tool_calls' in msg:
            for tc in msg['tool_calls']:
                tool_call_map[tc['id']] = tc['function']['name']
    
    for message in chat_history:
        role = message['role']
        content = message['content']
        
        if role == 'user':
            message_list.append(
                MemoryRecord(
                    message=BaseMessage.make_user_message(
                        role_name="user", 
                        content=content
                    ),
                    role_at_backend=OpenAIBackendRole.USER
                )
            )
        elif role == 'assistant':
            # Check if this assistant message has tool_calls
            tool_calls = message.get('tool_calls')
            if tool_calls:
                # Use FunctionCallingMessage for assistant messages with tool calls
                # Extract function name and arguments from first tool_call
                first_call = tool_calls[0]
                func_name = first_call.get('function', {}).get('name')
                args_str = first_call.get('function', {}).get('arguments', '{}')
                import json
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}

                base_msg = FunctionCallingMessage(
                    role_name="assistant",
                    role_type=RoleType.ASSISTANT,
                    content=content,
                    meta_dict={'tool_calls': tool_calls},
                    func_name=func_name,
                    args=args,
                    tool_call_id=first_call.get('id')
                )
            else:
                base_msg = BaseMessage.make_assistant_message(
                    role_name="assistant",
                    content=content
                )
            message_list.append(
                MemoryRecord(
                    message=base_msg,
                    role_at_backend=OpenAIBackendRole.ASSISTANT
                )
            )
        elif role == 'system':
            if not skip_system:
                message_list.append(
                    MemoryRecord(
                        message=BaseMessage.make_system_message(
                            role_name="System",
                            content=content
                        ),
                        role_at_backend=OpenAIBackendRole.SYSTEM
                    )
                )
        elif role == 'function':
            message_list.append(
                MemoryRecord(
                    message=BaseMessage(
                        role_name="function",
                        role_type=RoleType.DEFAULT,
                        content=content,
                        meta_dict=None
                    ),
                    role_at_backend=OpenAIBackendRole.FUNCTION
                )
            )
        elif role == 'tool':
            # Tool messages use FunctionCallingMessage with FUNCTION role
            tool_call_id = message.get('tool_call_id')
            func_name = message.get('func_name')

            # If func_name not provided, try to look it up from tool_call_map
            if not func_name and tool_call_id:
                func_name = tool_call_map.get(tool_call_id)

            message_list.append(
                MemoryRecord(
                    message=FunctionCallingMessage(
                        role_name="tool",
                        role_type=RoleType.DEFAULT,
                        content=content,
                        meta_dict=None,
                        result=content,
                        tool_call_id=tool_call_id,
                        func_name=func_name
                    ),
                    role_at_backend=OpenAIBackendRole.FUNCTION
                )
            )
        elif role == 'developer':
            message_list.append(
                MemoryRecord(
                    message=BaseMessage(
                        role_name="developer",
                        role_type=RoleType.DEFAULT,
                        content=content,
                        meta_dict=None
                    ),
                    role_at_backend=OpenAIBackendRole.DEVELOPER
                )
            )
        else:
            raise ValueError(f"Unsupported role: {role}.")
    
    return message_list



def memoryRecords_to_messages(
    records: List[MemoryRecord]
) -> List[dict]:
    """Convert MemoryRecord list to standard message format.
    
    Args:
        records: List of MemoryRecord objects.
    """
    return [record.to_openai_message() for record in records]

def add_tool_requests_to_chat_history(
    chat_history: List[dict],
    tool_request,
) -> List[dict]:
    """Add tool requests to chat history."""
    last_msg = chat_history[-1]
    if last_msg.get("role") == "assistant":
        # Format tool_calls according to what record_interaction expects
        # Arguments must be JSON string for messages_to_memoryRecords
        args = tool_request.args or {}
        args_str = json.dumps(args) if isinstance(args, dict) else args
        last_msg["tool_calls"] = [
            {
                "id": tool_request.tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_request.tool_name,
                    "arguments": args_str,
                },
            }
        ]
    return chat_history


@dataclass
class StepResult:
    """Result from ExternalToolAgent.step().

    Attributes:
        content: Final response text from the agent.
        num_tool_calls: Number of tool calls made.
        tools_used: List of unique tool names used.
        terminated_early: Whether execution stopped before natural completion.
        termination_reason: Reason for early termination (if any).
        usage: Accumulated token usage across all internal calls.
    """
    content: str
    num_tool_calls: int = 0
    tools_used: List[str] = field(default_factory=list)
    terminated_early: bool = False
    termination_reason: str = ""
    usage: Optional[dict] = None


class ExternalToolAgent:
    """Wrapper for ChatAgent with external tools and context limit handling.

    Executes tools externally with explicit control flow, checking context
    length between each iteration. Stops gracefully when approaching token limit.

    Example:
        >>> agent = ExternalToolAgent(
        ...     system_message="You are a helpful assistant.",
        ...     model=worker_model,
        ...     tools=worker_tools,
        ...     reserved_tokens=2048,
        ... )
        >>> result = agent.step("Search for information about X")
        >>> print(result.content, result.num_tool_calls)

    With live logging:
        >>> from rosetta.workflow.display import ConvLogger
        >>> logger = ConvLogger()
        >>> agent = ExternalToolAgent(..., logger=logger)
        >>> result = agent.step("Search for X")  # Shows live updates
    """

    def __init__(
        self,
        system_message: str,
        model: BaseModelBackend,
        tools: List[FunctionTool],
        reserved_tokens: int = 2048,
        token_limit: Optional[int] = None,
        logger: Optional[Any] = None,
    ):
        """Initialize ExternalToolAgent.

        Args:
            system_message: System prompt for the agent.
            model: Model backend to use.
            tools: List of tools available to the agent.
            reserved_tokens: Tokens to reserve; stops when limit approached.
            token_limit: Context window size. If None, defaults to 128000.
            logger: Optional ConvLogger for live message display. Must have
                start(), stop(), and update(messages) methods.
        """
        self.agent = ChatAgent(
            system_message=system_message,
            model=model,
            external_tools=tools,
            summarize_threshold=None,
        )
        self.tool_map = {tool.get_function_name(): tool for tool in tools}
        self.token_limit = token_limit if token_limit is not None else 128000
        self.token_threshold = self.token_limit - reserved_tokens
        self.logger = logger
        self._accumulated_usage: dict = {}

    def _accumulate_usage(self, response) -> None:
        """Accumulate token usage from a response.

        Args:
            response: ChatAgentResponse or raw model response with usage info.
        """
        usage = None
        if hasattr(response, 'info') and response.info:
            usage = response.info.get("usage")
        elif hasattr(response, 'usage') and response.usage:
            usage = response.usage

        if usage is None:
            return

        # Convert usage object to dict if needed
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif not isinstance(usage, dict):
            # Try extracting attributes
            usage_dict = {}
            for key in ["completion_tokens", "prompt_tokens", "total_tokens"]:
                if hasattr(usage, key):
                    usage_dict[key] = getattr(usage, key)
            usage = usage_dict if usage_dict else None

        if usage is None:
            return

        # Accumulate
        for key in ["completion_tokens", "prompt_tokens", "total_tokens"]:
            if key in usage:
                self._accumulated_usage[key] = self._accumulated_usage.get(key, 0) + usage[key]

    def _get_accumulated_usage(self) -> Optional[dict]:
        """Get accumulated usage and reset the accumulator."""
        if not self._accumulated_usage:
            return None
        result = dict(self._accumulated_usage)
        self._accumulated_usage = {}
        return result

    @property
    def memory(self):
        """Access underlying agent's memory."""
        return self.agent.memory

    @property
    def chat_history(self):
        """Access underlying agent's chat history."""
        return self.agent.chat_history

    def _generate_summary(self, task: str) -> str:
        """Generate summary when exiting due to token limit."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            result = self.agent.summarize(filename=None, include_summaries=False)
        summary = result.get("summary", "")
        if summary:
            return f"[Context limit - summary]\nTask: {task}\n\n{summary}"
        return f"[Context limit]\nTask: {task}\nUnable to generate summary."

    def _check_context_limit(self) -> bool:
        """Check if context length exceeds threshold.

        Returns:
            True if over threshold, False otherwise.
        """
        try:
            _, num_tokens = self.agent.memory.get_context()
            return num_tokens >= self.token_threshold
        except Exception:
            return False

    def _execute_tool(self, request) -> str:
        """Execute a tool request and return result."""
        tool = self.tool_map.get(request.tool_name)
        if tool is None:
            return f"Error: Tool '{request.tool_name}' not found"
        try:
            return tool(**request.args)
        except Exception as e:
            return f"Error executing tool '{request.tool_name}': {e}"

    def _continue_from_tool_result(self):
        """Continue conversation after tool execution without adding user message.

        Returns:
            ChatCompletion from the model (streaming handled transparently).
        """
        openai_messages, num_tokens = self.agent.memory.get_context()
        response = model_run_sync(self.agent.model_backend, openai_messages)

        # Update memory with assistant response
        from camel.messages import BaseMessage
        from camel.types import OpenAIBackendRole
        content = response.choices[0].message.content or ""
        assistant_msg = BaseMessage.make_assistant_message(role_name="assistant", content=content)
        self.agent.update_memory(assistant_msg, OpenAIBackendRole.ASSISTANT)

        return response

    def step(self, message: str, max_iterations: Optional[int] = None) -> StepResult:
        """Execute task with external tool handling.

        Runs until the agent completes (no tool call) or a limit is reached.
        If a logger is configured, shows live message updates during execution.

        Args:
            message: Initial message/task to send to the agent.
            max_iterations: Maximum number of tool calls. None for unlimited.

        Returns:
            StepResult with response content and execution metadata.
        """
        num_tool_calls = 0
        tools_used = []
        is_first_call = True
        self._accumulated_usage = {}  # Reset for this step call

        # Start live logging if logger is configured
        if self.logger:
            self.logger.start()

        def _finish(result: StepResult) -> StepResult:
            """Stop logger, add usage, and return result."""
            if self.logger:
                self.logger.stop()
            result.usage = self._get_accumulated_usage()
            return result

        while True:
            # Check context length before each call
            if self._check_context_limit():
                return _finish(StepResult(
                    content=self._generate_summary(message),
                    num_tool_calls=num_tool_calls,
                    tools_used=tools_used,
                    terminated_early=True,
                    termination_reason="Token limit reached. Summarized.",
                ))

            # Call agent
            try:
                if is_first_call:
                    response = self.agent.step(message)
                    self._accumulate_usage(response)
                    is_first_call = False
                else:
                    # Continue without adding user message
                    response = self._continue_from_tool_result()
                    self._accumulate_usage(response)
            except Exception as e:
                error_str = str(e).lower()
                if any(x in error_str for x in ["too long", "context length", "maximum context"]):
                    return _finish(StepResult(
                        content=self._generate_summary(message),
                        num_tool_calls=num_tool_calls,
                        tools_used=tools_used,
                        terminated_early=True,
                        termination_reason="Context exceeded. Summarized.",
                    ))
                if self.logger:
                    self.logger.stop()
                raise

            # Update logger after agent response
            if self.logger:
                self.logger.update(self.chat_history)

            # Check for external tool request
            if hasattr(response, 'info'):
                tool_requests = response.info.get("external_tool_call_requests", [])
            else:
                # Direct model response - check for tool_calls
                tool_calls = getattr(response.choices[0].message, 'tool_calls', None)
                if tool_calls:
                    # Build tool request from response
                    tc = tool_calls[0]
                    from camel.agents._types import ToolCallRequest
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except json.JSONDecodeError:
                            args = {}
                    tool_requests = [ToolCallRequest(
                        tool_name=tc.function.name,
                        args=args if isinstance(args, dict) else {},
                        tool_call_id=tc.id,
                    )]
                else:
                    tool_requests = []

            if not tool_requests:
                # No tool call - task complete
                content = response.msg.content if hasattr(response, 'msg') else response.choices[0].message.content
                return _finish(StepResult(
                    content=content or "",
                    num_tool_calls=num_tool_calls,
                    tools_used=tools_used,
                ))

            # Execute external tool
            request = tool_requests[0]
            if request.tool_name not in tools_used:
                tools_used.append(request.tool_name)

            result = self._execute_tool(request)

            # Fix the assistant message recorded by step() - it doesn't have
            # tool_calls properly encoded for the OpenAI API. We need to:
            # 1. Remove the incorrectly formatted assistant message
            # 2. Add a proper FunctionCallingMessage with tool_calls
            # 3. Add the tool result message
            self.agent.memory.pop_records(1)

            # Get assistant content from response
            if hasattr(response, 'msg'):
                assistant_content = response.msg.content or ""
            else:
                assistant_content = response.choices[0].message.content or ""

            # Add proper assistant message with tool_calls
            assist_msg = FunctionCallingMessage(
                role_name="assistant",
                role_type=RoleType.ASSISTANT,
                meta_dict=None,
                content=assistant_content,
                func_name=request.tool_name,
                args=request.args,
                tool_call_id=request.tool_call_id,
            )
            self.agent.update_memory(assist_msg, OpenAIBackendRole.ASSISTANT)

            # Add tool result message
            func_msg = FunctionCallingMessage(
                role_name="assistant",
                role_type=RoleType.ASSISTANT,
                meta_dict=None,
                content="",
                func_name=request.tool_name,
                result=result,
                tool_call_id=request.tool_call_id,
            )
            self.agent.update_memory(func_msg, OpenAIBackendRole.FUNCTION)

            # Update logger after tool execution
            if self.logger:
                self.logger.update(self.chat_history)

            num_tool_calls += 1

            # Check max iterations
            if max_iterations and num_tool_calls >= max_iterations:
                content = response.msg.content if hasattr(response, 'msg') else response.choices[0].message.content
                return _finish(StepResult(
                    content=content or "Max iterations reached",
                    num_tool_calls=num_tool_calls,
                    tools_used=tools_used,
                    terminated_early=True,
                    termination_reason="Max iterations reached.",
                ))