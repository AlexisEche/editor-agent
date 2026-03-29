# -*- coding: utf-8 -*-
"""
Cliente mínimo para Ollama (/api/chat) con herramientas, compatible con el flujo Bedrock del agente.

Requiere Ollama instalado y en ejecución (`ollama serve`), y el modelo descargado (`ollama pull ...`).
Sin coste de API: corre en local (RAM/CPU/GPU según el modelo).
"""

from __future__ import annotations

import json
import os
import uuid
from json import JSONDecoder
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lib.gemini_llm import GeminiAgentResponse, _TextPart, _ToolUsePart


def ollama_base_url() -> str:
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def ollama_model_id() -> str:
    # qwen2.5-coder suele seguir mejor instrucciones de código/tools que llama3.2; `ollama pull …` si no está.
    return os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b").strip() or "qwen2.5-coder:7b"


def ollama_timeout_s() -> float:
    return float(os.environ.get("OLLAMA_TIMEOUT", "900"))


def check_ollama_ready() -> None:
    """
    Verifica que el daemon responda y que el modelo configurado exista localmente.
    Falla con mensaje accionable si falta `ollama serve` o `ollama pull`.
    """
    if os.environ.get("OLLAMA_SKIP_READY_CHECK", "").lower() in ("1", "true", "yes"):
        return
    base = ollama_base_url()
    try:
        tags_req = Request(f"{base}/api/tags", method="GET")
        with urlopen(tags_req, timeout=5.0) as resp:
            resp.read()
    except URLError as e:
        raise RuntimeError(
            f"No se pudo conectar a Ollama en {base}. "
            "Instalá desde https://ollama.com, abrí la app (o ejecutá `ollama serve`) y reintentá."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Ollama en {base}: {e}") from e

    mid = ollama_model_id()
    payload = json.dumps({"name": mid}).encode("utf-8")
    show = Request(
        f"{base}/api/show",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(show, timeout=30.0) as resp:
            resp.read()
    except HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        if e.code == 404:
            raise RuntimeError(
                f"El modelo `{mid}` no está instalado en Ollama. "
                f"Ejecutá en una terminal: `ollama pull {mid}`"
            ) from e
        raise RuntimeError(f"Ollama /api/show HTTP {e.code}: {body}") from e
    except URLError as e:
        raise RuntimeError(
            f"Ollama no alcanzable al comprobar el modelo ({base}). ¿Está el servicio en marcha? ({e})"
        ) from e


def bedrock_tool_schemas_to_ollama(schemas: list[dict]) -> list[dict]:
    """Convierte esquemas estilo Bedrock/OpenAI function a tools de Ollama."""
    out: list[dict] = []
    for s in schemas:
        params = s.get("parameters") or {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "parameters": params,
                },
            }
        )
    return out


def bedrock_messages_to_ollama(messages: list[dict]) -> list[dict]:
    """Historial interno (user/assistant con toolUse/toolResult) → mensajes Ollama."""
    omsg: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        parts = msg.get("content") or []
        if role == "user":
            for p in parts:
                if "text" in p:
                    omsg.append({"role": "user", "content": p["text"]})
                elif "toolResult" in p:
                    tr = p["toolResult"]
                    name = tr.get("name") or "unknown"
                    tid = (tr.get("toolUseId") or "").strip()
                    payload: Any = {}
                    for b in tr.get("content") or []:
                        if "json" in b:
                            payload = b["json"]
                            break
                    body = json.dumps(payload, ensure_ascii=False)
                    tm: dict[str, Any] = {"role": "tool", "content": body}
                    if tid:
                        tm["tool_call_id"] = tid
                    if name:
                        tm["name"] = name
                    omsg.append(tm)
        elif role == "assistant":
            text_bits: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for p in parts:
                if "text" in p:
                    text_bits.append(p["text"])
                elif "toolUse" in p:
                    tu = p["toolUse"]
                    inp = tu.get("input")
                    if isinstance(inp, str):
                        try:
                            inp = json.loads(inp)
                        except json.JSONDecodeError:
                            inp = {"raw": inp}
                    elif inp is None:
                        inp = {}
                    cid = str(tu.get("toolUseId") or uuid.uuid4())
                    tool_calls.append(
                        {
                            "id": cid,
                            "type": "function",
                            "function": {
                                "name": tu["name"],
                                "arguments": inp,
                            },
                        }
                    )
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_bits) if text_bits else "",
            }
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            omsg.append(assistant)
    return omsg


def ollama_tool_use_system_reminder() -> str:
    """Anexar al system prompt: modelos locales a menudo “simulan” tools en markdown."""
    return (
        "\n\n## Herramientas (obligatorio — Ollama)\n"
        "Tenés que invocar las **funciones reales** del asistente (tool calling de la API). "
        "**Prohibido** responder solo con JSON en texto, bloques ```json, o claves `name`/`arguments` "
        "en el mensaje: eso **no ejecuta** nada en el sandbox. "
        "En cada turno donde haga falta actuar, emití llamadas de herramienta válidas "
        "(p. ej. `read_file`, `execute_code`, `write_file`). "
        "El sandbox **no tiene pantalla**: **tkinter/PyQt fallan**; el stack es **Next.js en `web-app/`** "
        "(React + CSS/Tailwind para look tipo Windows 95), no apps de escritorio. "
        "Primero **MVP que pase `npm run build`** (ideal: todo en `web-app/src/app/page.tsx`); "
        "`execute_code` puede ser solo la línea `npm run build` (se convierte a Python). "
        "No repitas el mismo `write_file` si ya se aplicó: seguí con **build** y corregí errores."
    )


def _is_tool_invocation_dict(obj: Any, allowed_names: set[str]) -> bool:
    if not isinstance(obj, dict) or obj.get("name") not in allowed_names:
        return False
    args = obj.get("arguments")
    return isinstance(args, dict) or isinstance(args, str)


def _dict_to_ollama_tool_call(obj: dict[str, Any]) -> dict[str, Any]:
    name = str(obj["name"])
    args = obj.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {"raw": args}
    elif isinstance(args, dict):
        pass
    else:
        args = {} if args is None else {"raw": args}
    return {
        "id": str(uuid.uuid4()),
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _coerce_tool_calls_from_assistant_text(
    content: str,
    allowed_names: set[str],
) -> list[dict[str, Any]]:
    """
    Varios modelos locales devuelven ```json {"name":"execute_code",...} ``` en lugar de tool_calls.
    Si el JSON es válido y el nombre coincide con una herramienta declarada, lo normalizamos.
    """
    if not content.strip() or not allowed_names:
        return []
    if os.environ.get("OLLAMA_COERCE_TOOLS_FROM_JSON", "1").lower() in ("0", "false", "no"):
        return []
    decoder = JSONDecoder()
    # Tras ```json o ```, primer {
    for marker in ("```json", "```JSON", "```"):
        p = content.find(marker)
        if p >= 0:
            brace = content.find("{", p + len(marker))
            if brace >= 0:
                try:
                    obj, _ = decoder.raw_decode(content, brace)
                    if _is_tool_invocation_dict(obj, allowed_names):
                        return [_dict_to_ollama_tool_call(obj)]
                except json.JSONDecodeError:
                    pass
    # Primer objeto JSON en el mensaje
    brace = content.find("{")
    while brace >= 0:
        try:
            obj, end = decoder.raw_decode(content, brace)
            if _is_tool_invocation_dict(obj, allowed_names):
                return [_dict_to_ollama_tool_call(obj)]
        except json.JSONDecodeError:
            pass
        brace = content.find("{", brace + 1)
    return []


def _assistant_text_without_fenced_json(content: str) -> str:
    """Quita un bloque ```json ... ``` inicial para no duplicar payload en el historial."""
    for marker in ("```json", "```JSON", "```"):
        p = content.find(marker)
        if p < 0:
            continue
        close = content.find("```", p + len(marker))
        if close >= 0:
            before = content[:p].strip()
            after = content[close + 3 :].strip()
            return (before + "\n" + after).strip()
    return content.strip()


def _post_chat(payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{ollama_base_url()}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=ollama_timeout_s()) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {err}") from e
    except URLError as e:
        raise RuntimeError(
            f"Ollama no alcanzable en {ollama_base_url()}. ¿Está `ollama serve` corriendo? ({e})"
        ) from e


def llm_ollama(
    _client: Any,
    messages: list[dict],
    system: str,
    tools: Optional[list[dict]] = None,
) -> GeminiAgentResponse:
    ollama_messages = bedrock_messages_to_ollama(messages)
    payload: dict[str, Any] = {
        "model": ollama_model_id(),
        "messages": ollama_messages,
        "stream": False,
        "options": {
            "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.2")),
        },
    }
    if system and system.strip():
        payload["system"] = system.strip()
    if tools:
        payload["tools"] = bedrock_tool_schemas_to_ollama(tools)

    raw = _post_chat(payload)
    omsg = raw.get("message") or {}
    content = omsg.get("content") or ""
    tool_calls: list[Any] = list(omsg.get("tool_calls") or [])
    allowed_tool_names = {t["name"] for t in tools} if tools else set()
    coerced = False
    if tools and not tool_calls and isinstance(content, str):
        coerced_list = _coerce_tool_calls_from_assistant_text(content, allowed_tool_names)
        if coerced_list:
            tool_calls = coerced_list
            coerced = True
            print(
                "[ollama] El modelo devolvió herramientas como texto JSON; se normalizaron a tool_calls.",
                flush=True,
            )

    bd_parts: list[dict[str, Any]] = []
    out: list[Any] = []
    text_for_ui = content if isinstance(content, str) else ""
    if coerced and isinstance(text_for_ui, str):
        text_for_ui = _assistant_text_without_fenced_json(text_for_ui)
        if not text_for_ui.strip():
            text_for_ui = ""
    if isinstance(text_for_ui, str) and text_for_ui.strip():
        bd_parts.append({"text": text_for_ui})
        out.append(_TextPart(text_for_ui))
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name") or "unknown"
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"raw": args}
        elif args is None:
            args = {}
        cid = str(tc.get("id") or uuid.uuid4())
        bd_parts.append({"toolUse": {"toolUseId": cid, "name": name, "input": args}})
        out.append(_ToolUsePart(name=name, arguments=json.dumps(args), call_id=cid))

    if not bd_parts:
        bd_parts.append({"text": "(sin texto ni herramientas en la respuesta de Ollama)"})
        out.append(_TextPart(bd_parts[0]["text"]))

    bedrock_msg = {"role": "assistant", "content": bd_parts}
    return GeminiAgentResponse(bedrock_msg, out)


def ollama_text_only(_client: Any, user_text: str, system: str) -> str:
    payload: dict[str, Any] = {
        "model": ollama_model_id(),
        "messages": [{"role": "user", "content": user_text}],
        "stream": False,
        "options": {"temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))},
    }
    if system and system.strip():
        payload["system"] = system.strip()
    raw = _post_chat(payload)
    omsg = raw.get("message") or {}
    c = omsg.get("content")
    return (c if isinstance(c, str) else str(c)).strip()
