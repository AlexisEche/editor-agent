# -*- coding: utf-8 -*-
"""
Conteo de tokens y compresión del historial (**Runtime Summary**, Tarea Final).

- Tope por defecto 40_000 tokens; al superarlo se resume el **70 %** de mensajes
  más antiguos y se reemplazan por un par resumen (user/assistant) + recientes.
- Umbral configurable con `MAX_CONTEXT_TOKENS_BEFORE_COMPRESS` (mismo valor que `MAX_TOKENS`).
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional, Protocol


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


# Enunciado Tarea Final: MAX_TOKENS = 40_000, COMPRESSION_RATIO = 0.70
MAX_TOKENS = _env_int("MAX_CONTEXT_TOKENS_BEFORE_COMPRESS", 40_000)
COMPRESSION_RATIO = 0.70


class Summarizer(Protocol):
    def __call__(self, transcript: str) -> str: ...


def count_tokens(messages: list) -> int:
    """Cuenta tokens aproximados del historial (encoding cl100k_base si hay tiktoken)."""
    blob = json.dumps(messages, ensure_ascii=False, default=str)
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(blob))
    except ImportError:
        return max(1, len(blob) // 4)


def _messages_preview(messages: list) -> str:
    parts: list[str] = []
    for i, m in enumerate(messages):
        parts.append(f"--- mensaje {i} ---\n{json.dumps(m, ensure_ascii=False, default=str)}")
    return "\n".join(parts)


def compress_context(messages: list, llm_client: Callable[[str], str]) -> list:
    """
    Comprime el COMPRESSION_RATIO de mensajes más antiguos vía LLM.
    Retorna: [resumen_user, resumen_assistant] + msgs_recientes
    """
    if not messages:
        return messages

    n = len(messages)
    k = max(1, int(n * COMPRESSION_RATIO))
    if k >= n:
        k = max(1, n - 1)

    old = messages[:k]
    recent = messages[k:]
    transcript = _messages_preview(old)
    cap = _env_int("MAX_COMPRESS_INPUT_CHARS", 100_000)
    if len(transcript) > cap:
        transcript = transcript[:cap] + "\n...[truncado por MAX_COMPRESS_INPUT_CHARS]...\n"

    prompt = (
        "Resumí con fidelidad técnica (herramientas, archivos, errores, pendientes). "
        "Máximo ~400 palabras. Español.\n\n"
        + transcript
    )
    summary = llm_client(prompt)

    summary_user = {
        "role": "user",
        "content": [
            {
                "text": (
                    f"[Resumen comprimido de los {k} mensajes más antiguos del historial]\n\n"
                    f"{summary}"
                )
            }
        ],
    }
    summary_assistant = {
        "role": "assistant",
        "content": [
            {
                "text": (
                    "He incorporado el resumen del contexto anterior. "
                    "Sigo con el estado actual del proyecto y las instrucciones recientes."
                )
            }
        ],
    }
    return [summary_user, summary_assistant] + recent


def maybe_compress(
    messages: list,
    llm_client: Callable[[str], str],
    *,
    max_tokens_before_compress: Optional[int] = None,
) -> list:
    """
    Solo comprime si el historial supera el tope.
    Si ``max_tokens_before_compress`` es None, usa ``MAX_CONTEXT_TOKENS_BEFORE_COMPRESS`` (default 40_000).
    Para Groq free (TPM ~6000 por request incl. system+tools) u **Ollama** (contexto local limitado),
    pasá un tope bajo desde el agente vía ``max_tokens_before_compress``.
    """
    flag = os.environ.get("DISABLE_CONTEXT_COMPRESSION", "").lower()
    if flag in ("1", "true", "yes"):
        return messages
    if max_tokens_before_compress is not None:
        limit = max(500, max_tokens_before_compress)
    else:
        limit = _env_int("MAX_CONTEXT_TOKENS_BEFORE_COMPRESS", 40_000)
    if count_tokens(messages) <= limit:
        return messages
    return compress_context(messages, llm_client)
