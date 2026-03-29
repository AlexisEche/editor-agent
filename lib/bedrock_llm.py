# -*- coding: utf-8 -*-
"""Cliente Bedrock: Converse (tools) + InvokeModel para texto Anthropic (menos overhead en resúmenes)."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

# Nova Pro (default del curso). Alternativa: Claude Haiku vía BEDROCK_MODEL_ID
DEFAULT_MODEL_ID = "amazon.nova-pro-v1:0"


def bedrock_model_id() -> str:
    return os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID).strip()


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _is_anthropic_model(model_id: str) -> bool:
    return model_id.startswith("anthropic.")


def _cap_max_output_tokens(model_id: str, requested: int) -> int:
    """
    Bedrock rechaza Converse si maxTokens supera el tope del modelo.
    Claude 3 Haiku: límite 4096; la API exige estrictamente **menor** que 4096 → máx 4095.
    """
    m = model_id.lower()
    if "haiku" in m:
        return max(1, min(requested, 4095))
    if m.startswith("anthropic."):
        return max(1, min(requested, 8192))
    return max(1, min(requested, 8192))


def _converse_inference_config() -> dict[str, Any]:
    """Limita tokens de salida y temperatura (reduce costo y respuestas enormes)."""
    mid = bedrock_model_id()
    requested = _env_int("BEDROCK_MAX_TOKENS", 8192)
    return {
        "maxTokens": _cap_max_output_tokens(mid, requested),
        "temperature": _env_float("BEDROCK_TEMPERATURE", 0.2),
    }


def schema_to_bedrock(schema: dict) -> dict:
    return {
        "toolSpec": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "inputSchema": {"json": schema.get("parameters", {})},
        }
    }


def llm(client, messages: list, system: str, tools: Optional[list] = None) -> "_BedrockResponse":
    kwargs: dict[str, Any] = dict(
        modelId=bedrock_model_id(),
        system=[{"text": system}],
        messages=messages,
        inferenceConfig=_converse_inference_config(),
    )
    if tools:
        kwargs["toolConfig"] = {"tools": [schema_to_bedrock(t) for t in tools]}

    raw = client.converse(**kwargs)
    return _BedrockResponse(raw)


def _invoke_anthropic_messages(
    client,
    *,
    user_text: str,
    system: str,
    max_tokens: int,
) -> str:
    """InvokeModel con formato Messages API (mismo patrón que projecto_ejemplo)."""
    mid = bedrock_model_id()
    max_tokens = _cap_max_output_tokens(mid, max_tokens)
    body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": _env_float("BEDROCK_SUMMARY_TEMPERATURE", 0.0),
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
    }
    if system and system.strip():
        body["system"] = system.strip()

    resp = client.invoke_model(
        modelId=mid,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(resp["body"].read())
    parts = payload.get("content") or []
    if not parts:
        return ""
    return (parts[0].get("text") or "").strip()


class _BedrockResponse:
    def __init__(self, raw: dict) -> None:
        self._raw = raw
        self.output: list = []
        self.output_text = ""
        self._bedrock_msg = raw["output"]["message"]
        self._parse()

    def _parse(self) -> None:
        for block in self._bedrock_msg.get("content", []):
            if "text" in block:
                self.output.append(_TextPart(block["text"]))
                self.output_text += block["text"]
            elif "toolUse" in block:
                tu = block["toolUse"]
                self.output.append(
                    _ToolUsePart(
                        name=tu["name"],
                        arguments=json.dumps(tu["input"]),
                        call_id=tu["toolUseId"],
                    )
                )

    def bedrock_message(self) -> dict:
        return self._bedrock_msg


class _TextPart:
    type = "message"

    def __init__(self, text: str) -> None:
        self.text = text
        self.content = text


class _ToolUsePart:
    type = "function_call"

    def __init__(self, name: str, arguments: str, call_id: str) -> None:
        self.name = name
        self.arguments = arguments
        self.call_id = call_id


def converse_text_only(client, user_text: str, system: str) -> str:
    """
    Llamada sin herramientas (p. ej. resúmenes).
    Para modelos Anthropic usa InvokeModel con tope bajo de salida (como projecto_ejemplo).
    Para el resto, Converse con inferenceConfig.
    """
    mid = bedrock_model_id()
    if _is_anthropic_model(mid) and os.environ.get("BEDROCK_TEXT_USE_CONVERSE", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        cap = _env_int("BEDROCK_SUMMARY_MAX_TOKENS", 1200)
        return _invoke_anthropic_messages(client, user_text=user_text, system=system, max_tokens=cap)

    messages = [{"role": "user", "content": [{"text": user_text}]}]
    resp = llm(client, messages, system, tools=None)
    return resp.output_text.strip()
