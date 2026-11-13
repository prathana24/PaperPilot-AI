"""
Session-Scoped Conversation Memory Manager for PaperPilot-AI.
Maintains isolated, document-scoped conversation histories across Streamlit sessions
with token budgeting and history trimming.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class ChatTurn:
    """Represents a single conversational turn."""
    role: str  # 'user' or 'assistant'
    content: str
    intent: Optional[str] = None
    agent: Optional[str] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "intent": self.intent,
            "agent": self.agent,
            "sources": self.sources,
            "timestamp": self.timestamp,
        }


class ConversationMemory:
    """Manages memory for a specific document within a session."""

    def __init__(self, document_id: str, max_turns: int = 10):
        self.document_id: str = document_id
        self.max_turns: int = max_turns
        self.history: List[ChatTurn] = []

    def add_user_message(self, content: str) -> None:
        """Appends a user message to the conversation."""
        if content and content.strip():
            self.history.append(ChatTurn(role="user", content=content.strip()))
            self._trim_history()

    def add_assistant_message(
        self,
        content: str,
        intent: Optional[str] = None,
        agent: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Appends an assistant response with metadata."""
        if content and content.strip():
            self.history.append(
                ChatTurn(
                    role="assistant",
                    content=content.strip(),
                    intent=intent,
                    agent=agent,
                    sources=sources or [],
                )
            )
            self._trim_history()

    def get_recent_history(self, limit: Optional[int] = None) -> List[ChatTurn]:
        """Returns the most recent conversation turns."""
        k = limit if limit is not None else self.max_turns
        return self.history[-k:]

    def get_formatted_history(self, limit: int = 6) -> str:
        """Formats recent history into a concise text block for LLM prompts."""
        recent = self.get_recent_history(limit=limit)
        if not recent:
            return ""

        formatted_lines = []
        for turn in recent:
            prefix = "User" if turn.role == "user" else "Assistant"
            # Truncate very long past turns to stay within token budgets
            preview = turn.content[:350] + ("..." if len(turn.content) > 350 else "")
            formatted_lines.append(f"{prefix}: {preview}")
        return "\n".join(formatted_lines)

    def clear(self) -> None:
        """Clears all conversation turns for this document."""
        self.history.clear()

    def _trim_history(self) -> None:
        """Keeps history within max_turns limit."""
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]


class SessionMemoryManager:
    """Manages document-scoped memories across multiple sessions and documents."""

    def __init__(self, default_max_turns: int = 10):
        self.default_max_turns = default_max_turns
        # Key: (session_id, document_id) -> ConversationMemory
        self._stores: Dict[str, ConversationMemory] = {}

    def _make_key(self, session_id: str, document_id: str) -> str:
        return f"{session_id}::{document_id}"

    def get_memory(self, session_id: str, document_id: str) -> ConversationMemory:
        """Retrieves or creates an isolated conversation store for the session and document."""
        key = self._make_key(session_id, document_id)
        if key not in self._stores:
            self._stores[key] = ConversationMemory(
                document_id=document_id,
                max_turns=self.default_max_turns
            )
        return self._stores[key]

    def clear_document_memory(self, session_id: str, document_id: str) -> None:
        """Clears the history for a specific document in a specific session."""
        key = self._make_key(session_id, document_id)
        if key in self._stores:
            self._stores[key].clear()

    def clear_session(self, session_id: str) -> None:
        """Clears all documents associated with a session."""
        keys_to_delete = [k for k in self._stores.keys() if k.startswith(f"{session_id}::")]
        for k in keys_to_delete:
            del self._stores[k]


# Global memory manager instance
_GLOBAL_MEMORY_MANAGER = SessionMemoryManager()


def get_session_memory(session_id: str, document_id: str) -> ConversationMemory:
    """Helper to access session memory."""
    return _GLOBAL_MEMORY_MANAGER.get_memory(session_id, document_id)
