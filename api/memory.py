  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
"""
Simple conversational memory store for Aegis agent usage.

This is intentionally minimal:
- ConversationMemory keeps the last N messages in memory (circular buffer).
- It supports get_conversation() which returns assembled history for prompt injection,
  and add() to append new messages.

In production, replace with:
- persistent conversation DB (Postgres) or vector DB (for retrieval-augmented generation),
- a memory policy/summary mechanism (evict, summarize, or chunk).
"""
from typing import List, Dict, Any
import threading
import time
import logging

logger = logging.getLogger("aegis.memory")


class ConversationMemory:
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._msgs: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(self, message: Dict[str, Any]):
        """
        message: {"role": "user"|"assistant"|"tool", "content": "..."}
        """
        with self._lock:
            self._msgs.append({"ts": time.time(), **message})
            if len(self._msgs) > self.capacity:
                # drop oldest
                self._msgs = self._msgs[-self.capacity :]

    def get_conversation(self, max_chars: int = 2000) -> str:
        with self._lock:
            out = []
            for m in self._msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                out.append(f"[{role}] {content}")
            convo = "\n".join(out)
            # trim if too long
            if len(convo) > max_chars:
                return convo[-max_chars:]
            return convo

    def clear(self):
        with self._lock:
            self._msgs = []
