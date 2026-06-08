"""LangGraph-based AI optimization-review adapter (see reviewer.py).

The optional ``langgraph``/``langchain-openai`` dependencies are only
imported lazily, inside :class:`AIReviewer`, so this package stays
importable when the ``[ai]`` extras are not installed.
"""
