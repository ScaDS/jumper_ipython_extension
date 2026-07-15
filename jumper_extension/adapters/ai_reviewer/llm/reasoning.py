import re

# Reasoning models emit chain-of-thought inline, wrapped in <think>...</think>
# (or MiniMax's <mm:think>...</mm:think>). The vLLM endpoint we target does not
# split it into a separate reasoning field, so it arrives in the reply content.
_THINK_RE = re.compile(r"<(?:mm:)?think>(.*?)</(?:mm:)?think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<(?:mm:)?think>", re.IGNORECASE)


def _content_to_text(content: str | list) -> str:
    """Flatten LangChain message content (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def split_reasoning(content: str | list) -> tuple[str, str]:
    """Separate a model reply into ``(visible_text, reasoning)``.

    Removes every ``<think>``/``<mm:think>`` block from the reply and returns the
    concatenated reasoning separately. An unterminated block (e.g. a stream cut
    short before the closing tag) is dropped from the opening tag onward so a
    dangling thought never leaks into the report.
    """
    text = _content_to_text(content)
    reasoning_parts = _THINK_RE.findall(text)
    visible = _THINK_RE.sub("", text)

    open_tag = _OPEN_THINK_RE.search(visible)
    if open_tag:
        reasoning_parts.append(visible[open_tag.end():])
        visible = visible[:open_tag.start()]

    reasoning = "\n".join(part.strip() for part in reasoning_parts if part.strip())
    return visible.strip(), reasoning
