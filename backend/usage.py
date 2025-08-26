# usage.py
from dataclasses import dataclass

@dataclass
class Usage:
    chat_calls: int = 0
    embed_calls: int = 0

USAGE = Usage()
