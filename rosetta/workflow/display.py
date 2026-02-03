import time
from typing import Optional, List
from contextlib import contextmanager
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text
from rich.live import Live

# Spinner frames for animating current task
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class ConvLogger:
    """Live conversation display that updates in place."""

    _ROLE_STYLES = {
        "system": ("bold magenta", "System"),
        "user": ("bold green", "User"),
        "assistant": ("bold blue", "Assistant"),
        "tool": ("bold yellow", "Tool"),
    }

    def __init__(self, tokenizer=None, enabled: bool = True, max_content_len: int = 300, max_messages: int = 4, transient: bool = False):
        """Initialize ConvLogger.

        Args:
            tokenizer: Tokenizer for counting tokens.
            enabled: Whether to enable display.
            max_content_len: Max characters to show per message.
            max_messages: Max messages to display at once.
            transient: If True, clears display when stopped.
        """
        self.console = Console() if enabled else None
        self.tokenizer = tokenizer
        self.max_content_len = max_content_len
        self.max_messages = max_messages
        self.transient = transient
        self._live: Optional[Live] = None
        self._last_messages: List[dict] = []

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tokenizer, or estimate by chars."""
        if not text:
            return 0
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4  # Rough estimate

    def _count_message_tokens(self, msg: dict) -> dict:
        """Count tokens for all parts of a message.

        Returns:
            Dict with 'content', 'reasoning', 'tool_calls', 'total' token counts.
        """
        counts = {"content": 0, "reasoning": 0, "tool_calls": 0, "total": 0}

        # Content tokens
        content = msg.get("content", "")
        counts["content"] = self._count_tokens(content)

        # Reasoning tokens (some models return this separately)
        reasoning = msg.get("reasoning_content") or msg.get("reasoning", "")
        counts["reasoning"] = self._count_tokens(reasoning)

        # Tool calls tokens (count the function name and arguments)
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "")
                # Count name and arguments
                counts["tool_calls"] += self._count_tokens(name)
                counts["tool_calls"] += self._count_tokens(args)

        counts["total"] = counts["content"] + counts["reasoning"] + counts["tool_calls"]
        return counts

    def _shorten(self, text: Optional[str]) -> str:
        if not text:
            return "[dim](empty)[/dim]"
        s = " ".join(text.strip().split())
        if len(s) > self.max_content_len:
            return escape(s[: self.max_content_len - 1]) + "[dim]…[/dim]"
        return escape(s)

    def _format_message(self, msg: dict, idx: int) -> Text:
        """Format a single message as Rich Text."""
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning", "")
        style, label = self._ROLE_STYLES.get(role, ("white", role.capitalize()))

        # Count tokens with breakdown
        counts = self._count_message_tokens(msg)
        token_parts = [f"{counts['total']} tokens"]
        details = []
        if counts["reasoning"] > 0:
            details.append(f"reasoning: {counts['reasoning']}")
        if counts["tool_calls"] > 0:
            details.append(f"tools: {counts['tool_calls']}")
        if details:
            token_parts.append(f"({', '.join(details)})")
        token_str = " ".join(token_parts)
        lines = [f"[{style}][{idx}] {label}[/{style}] [dim]| {token_str}[/dim]"]

        # Show reasoning content if present
        if reasoning:
            lines.append(f"  [dim italic]<think> {self._shorten(reasoning)}[/dim italic]")

        # Show content
        if content:
            lines.append(f"  {self._shorten(content)}")

        # Show tool calls for assistant
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                if len(args) > 80:
                    args = args[:77] + "..."
                lines.append(f"  [dim]→ {name}({args})[/dim]")

        # Show tool_call_id for tool messages
        if msg.get("tool_call_id"):
            lines.append(f"  [dim]tool_call_id: {msg['tool_call_id'][:20]}...[/dim]")

        return Text.from_markup("\n".join(lines))

    def _render_all(self, messages: List[dict]) -> Group:
        """Render all messages as a Rich Group."""
        renderables = []

        # Calculate total token count with breakdown
        totals = {"content": 0, "reasoning": 0, "tool_calls": 0, "total": 0}
        for msg in messages:
            counts = self._count_message_tokens(msg)
            for k in totals:
                totals[k] += counts[k]

        header_parts = [f"[bold cyan]Total: {totals['total']} tokens[/bold cyan]"]
        details = []
        if totals["reasoning"] > 0:
            details.append(f"reasoning: {totals['reasoning']}")
        if totals["tool_calls"] > 0:
            details.append(f"tools: {totals['tool_calls']}")
        if details:
            header_parts.append(f"[dim]({', '.join(details)})[/dim]")
        header_parts.append(f"[dim]| {len(messages)} messages[/dim]")
        header = Text.from_markup(" ".join(header_parts))
        renderables.append(header)
        
        # Only show the latest N messages
        display_messages = messages[-self.max_messages:] if len(messages) > self.max_messages else messages
        start_idx = len(messages) - len(display_messages)
        for i, msg in enumerate(display_messages, start=start_idx):
            renderables.append(self._format_message(msg, i))
        return Group(*renderables)

    def start(self) -> None:
        """Start live display mode."""
        if self.console and self._live is None:
            self._live = Live(console=self.console, refresh_per_second=4, transient=self.transient)
            self._live.start()

    def stop(self) -> None:
        """Stop live display mode (final state remains visible)."""
        if self._live:
            self._live.stop()
            self._live = None

    def update(self, messages: List[dict]) -> None:
        """Update display with current messages (clears old, prints new in place)."""
        if not self.console:
            return

        self._last_messages = list(messages)

        if self._live:
            # Live mode: update in place
            self._live.update(self._render_all(messages))
        else:
            # Non-live mode: just print all
            self.print_all(messages)

    def reset(self) -> None:
        """Reset state."""
        self._last_messages = []

    def print_all(self, messages: List[dict]) -> None:
        """Print all messages (non-live, permanent output)."""
        if not self.console:
            return

        # Calculate total token count with breakdown
        totals = {"content": 0, "reasoning": 0, "tool_calls": 0, "total": 0}
        for msg in messages:
            counts = self._count_message_tokens(msg)
            for k in totals:
                totals[k] += counts[k]

        header_parts = [f"[bold cyan]Total: {totals['total']} tokens[/bold cyan]"]
        details = []
        if totals["reasoning"] > 0:
            details.append(f"reasoning: {totals['reasoning']}")
        if totals["tool_calls"] > 0:
            details.append(f"tools: {totals['tool_calls']}")
        if details:
            header_parts.append(f"[dim]({', '.join(details)})[/dim]")
        header_parts.append(f"[dim]| {len(messages)} messages[/dim]")
        header = Text.from_markup(" ".join(header_parts))
        self.console.print(header)
        
        # Only show the latest N messages
        display_messages = messages[-self.max_messages:] if len(messages) > self.max_messages else messages
        start_idx = len(messages) - len(display_messages)
        for i, msg in enumerate(display_messages, start=start_idx):
            self.console.print(self._format_message(msg, i))


class SubagentDisplayProxy:
    """Proxy that ExternalToolAgent uses to update StatusLogger's subagent display."""

    def __init__(self, update_callback):
        self._update_callback = update_callback

    def start(self) -> None:
        """No-op: StatusLogger manages the Live display."""
        pass

    def stop(self) -> None:
        """No-op: StatusLogger manages the Live display."""
        pass

    def update(self, messages: List[dict]) -> None:
        """Forward chat history updates to StatusLogger."""
        self._update_callback(messages)


class StatusLogger:
    """Handles console status display with spinner and history."""

    _ROLE_STYLES = {
        "system": ("bold magenta", "Sys"),
        "user": ("bold green", "User"),
        "assistant": ("bold blue", "Asst"),
        "tool": ("bold yellow", "Tool"),
    }

    def __init__(self, enabled: bool = True, max_tasks_preview: Optional[int] = None,
                 max_subagent_messages: int = 6, max_content_len: int = 200):
        self.console = Console() if enabled else None
        self.max_tasks_preview = max_tasks_preview
        self.max_subagent_messages = max_subagent_messages
        self.max_content_len = max_content_len

    @staticmethod
    def _shorten(text: Optional[str], max_len: int = 240) -> Optional[str]:
        if not text:
            return None
        s = " ".join(text.strip().split())
        return (s[: max_len - 1] + "…") if len(s) > max_len else s

    @staticmethod
    def _fmt_task(task: str, marker: str, dim: bool = False, is_spinner: bool = False) -> str:
        """Format a task with marker (✓, •, spinner, or space)."""
        t = escape(task.strip())
        if dim:
            return f"[dim]  {marker} {t}[/dim]"
        if is_spinner:
            return f"  [bold yellow]{marker}[/bold yellow] [bold]{t}[/bold]"
        return f"  {marker} {t}"

    @staticmethod
    def _get_spinner_frame() -> str:
        """Get current spinner frame based on time."""
        # Use time to cycle through frames (~10 frames per second)
        frame_idx = int(time.time() * 10) % len(SPINNER_FRAMES)
        return SPINNER_FRAMES[frame_idx]

    def _format_subagent_message(self, msg: dict, idx: int) -> str:
        """Format a single subagent message."""
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        style, label = self._ROLE_STYLES.get(role, ("white", role[:4].capitalize()))

        # Shorten content
        short_content = self._shorten(content, self.max_content_len) or "[dim](empty)[/dim]"

        lines = [f"    [{style}][{idx}] {label}[/{style}] {escape(short_content) if short_content != '[dim](empty)[/dim]' else short_content}"]

        # Show tool calls for assistant
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                if len(args) > 60:
                    args = args[:57] + "..."
                lines.append(f"      [dim]→ {escape(name)}({escape(args)})[/dim]")

        return "\n".join(lines)

    def _format_subagent_panel(self, messages: List[dict]) -> str:
        """Format subagent chat history as a panel."""
        if not messages:
            return ""

        lines = [f"  [bold cyan]Subagent Chat:[/bold cyan] [dim]({len(messages)} messages)[/dim]"]

        # Show latest N messages
        display_messages = messages[-self.max_subagent_messages:] if len(messages) > self.max_subagent_messages else messages
        start_idx = len(messages) - len(display_messages)

        for i, msg in enumerate(display_messages, start=start_idx):
            lines.append(self._format_subagent_message(msg, i))

        return "\n".join(lines)

    def _format_status(
        self,
        action: str,
        round_idx: int,
        step_idx: int,
        status_desc: str,
        finished: List[str],
        current: List[str],
        pending: List[str],
        response: Optional[str] = None,
        done: bool = False,
        tools_used: Optional[list] = None,
        subagent_messages: Optional[List[dict]] = None,
    ) -> str:
        """Format display content: ACTION | Round X | Step Y | Progress.

        Task markers:
            ✓ - finished tasks
            • - current task (in progress)
            ○ - pending tasks (todo)
        """
        progress = len(finished)
        total = len(finished) + len(current) + len(pending)

        checkmark = "[bold green]✓[/bold green] " if done else ""
        lines = [
            f"{checkmark}[bold green]{escape(action.capitalize())}[/bold green] [dim]|[/dim] "
            f"Round {round_idx} [dim]|[/dim] "
            f"Step {step_idx} [dim]|[/dim] "
            f"Progress {progress}/{total}",
            f"  [bold cyan]Status:[/bold cyan] {escape(status_desc)}",
        ]

        # Show tools used
        if tools_used:
            tools_str = ", ".join(escape(t) for t in tools_used)
            lines.append(f"  [bold cyan]Tools:[/bold cyan] {tools_str}")

        # Show subagent chat history (only while working)
        if not done and subagent_messages:
            lines.append(self._format_subagent_panel(subagent_messages))

        # Show response
        short = self._shorten(response)
        if short:
            lines.append(f"  [bold cyan]Response:[/bold cyan] {escape(short)}")

        # Show task lists only while working (not when done)
        if not done and (finished or current or pending):
            lines.append("  [bold cyan]Tasks:[/bold cyan]")
            for task in finished:
                lines.append(self._fmt_task(task, "[green]✓[/green]", dim=True))
            for task in current:
                # Use animated spinner for current task
                spinner = self._get_spinner_frame()
                lines.append(self._fmt_task(task, spinner, dim=False, is_spinner=True))
            display_pending = pending
            extra_count = 0
            if self.max_tasks_preview is not None and len(pending) > self.max_tasks_preview:
                display_pending = pending[:self.max_tasks_preview]
                extra_count = len(pending) - self.max_tasks_preview
            for task in display_pending:
                lines.append(self._fmt_task(task, "[dim]○[/dim]", dim=True))
            if extra_count > 0:
                lines.append(f"[dim]    (+{extra_count} more)[/dim]")

        return "\n".join(lines)

    @contextmanager
    def status(
        self,
        action: str,
        round_idx: int,
        step_idx: int,
        status_desc: str,
        finished: List[str],
        current: List[str],
        pending: List[str],
        show_subagent: bool = False,
    ):
        """Context manager: show spinner; yield updater for current subagent response.

        Args:
            show_subagent: If True, use Live display to show subagent chat history.
        """
        last_response: Optional[str] = None
        last_tools: Optional[list] = None
        last_tasks: tuple = (list(finished), list(current), list(pending))
        last_subagent_messages: List[dict] = []

        if not self.console:
            yield lambda *a, **kw: None, None
            return

        def make_update_response(updater):
            """Create the update_response callback."""
            def update_response(
                resp: Optional[str],
                tools_used: Optional[list] = None,
                tasks: Optional[tuple] = None,
            ):
                nonlocal last_response, last_tools, last_tasks
                last_response = resp
                last_tools = tools_used
                if tasks:
                    last_tasks = tasks
                updater(self._format_status(
                    action, round_idx, step_idx, status_desc, *last_tasks,
                    response=resp, tools_used=tools_used,
                    subagent_messages=last_subagent_messages if show_subagent else None
                ))
            return update_response

        def update_subagent(messages: List[dict]):
            """Update subagent chat history."""
            nonlocal last_subagent_messages
            last_subagent_messages = list(messages)

        if show_subagent:
            # Use Live display for subagent chat history
            status_msg = self._format_status(action, round_idx, step_idx, status_desc, *last_tasks)
            with Live(Text.from_markup(status_msg), console=self.console, refresh_per_second=4, transient=True) as live:
                def live_updater(msg):
                    live.update(Text.from_markup(msg))

                update_response = make_update_response(live_updater)
                subagent_proxy = SubagentDisplayProxy(lambda msgs: (update_subagent(msgs), live_updater(self._format_status(
                    action, round_idx, step_idx, status_desc, *last_tasks,
                    response=last_response, tools_used=last_tools,
                    subagent_messages=msgs
                ))))
                yield update_response, subagent_proxy

            # Print final status
            done_msg = self._format_status(
                action, round_idx, step_idx, status_desc, *last_tasks,
                response=last_response, done=True, tools_used=last_tools
            )
            self.console.print(done_msg, highlight=False)
        else:
            # Use simple spinner (no subagent display)
            status_msg = self._format_status(action, round_idx, step_idx, status_desc, *last_tasks)
            with self.console.status(status_msg) as st:
                update_response = make_update_response(st.update)
                yield update_response, None

            done_msg = self._format_status(
                action, round_idx, step_idx, status_desc, *last_tasks,
                response=last_response, done=True, tools_used=last_tools
            )
            self.console.print(done_msg, highlight=False)
