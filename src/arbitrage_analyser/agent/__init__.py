"""AI agent that answers questions about the funds in the database (Phase 1).

tools.py    read-only tools over the fund database
prompt.py   the agent's instructions
agent.py    the tool-use loop (works with any provider)
providers.py  AI model providers: Gemini (default), Claude, other OpenAI-compatible APIs
store.py    conversation storage (data/chat.db)
tracing.py  optional Langfuse tracing
"""
