# -*- coding: utf-8 -*-
"""
Groq OpenAI-compatible API (chat completions + tools).
Clave gratis en https://console.groq.com — cupo separado de Google Gemini.

Útil cuando AI Studio muestra limit:0 en free tier para los modelos Gemini.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import time
import uuid
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lib.gemini_llm import GeminiAgentResponse, _TextPart, _ToolUsePart

# Llama vía Groq a veces “simula” tools con XML (<function=...>); la API responde 400 tool_use_failed.
_GROQ_TOOL_FORMAT_RULES = """

[Instrucción de formato — Groq / OpenAI tool_calls]
Invocá herramientas **solo** con el mecanismo nativo de function calling (el servidor registra `tool_calls`). Está **terminantemente prohibido** escribir en el texto de la respuesta:
- líneas que contengan `<function=` o `</function>`
- XML o markdown que copie el nombre de una herramienta (ej. `<function=glob>`)
Eso **siempre** produce error `tool_use_failed` y no ejecuta nada.
Para `glob`, `list_directory`, etc., la invocación correcta es **solo** vía tool_calls del API, con `arguments` = JSON objeto válido (ej. glob: {"pattern": "**/*.tsx", "root": "web-app"}).
Para `execute_code` el único argumento es **"code"** (string) con el script Python completo, por ejemplo:
{"code": "import os, subprocess\\nsubprocess.run([\\"npx\\", ...], cwd=\\"web-app\\", ...)"}
**No** pongas el código Python como texto suelto donde iría un objeto JSON (evitá cosas como `{"import os` …).
"""

_GROQ_TOOL_FAIL_RETRY_NUDGE = (
    "[Reintento automático] Volviste a usar formato XML o inválido. "
    "NO escribas <function=glob> ni similar. Dejá el cuerpo del mensaje vacío o solo texto breve; "
    "las llamadas deben ser tool_calls nativos con JSON (glob: {\"pattern\":\"...\",\"root\":\"...\"})."
)

# Pausa mínima tras completar un request Groq (éxito o error) para no saturar TPM free. 0 = desactivado.
_last_groq_request_done: float = 0.0


def _groq_pace_before_request() -> None:
    global _last_groq_request_done
    gap = float(os.environ.get("GROQ_MIN_SECONDS_BETWEEN_REQUESTS", "5"))
    if gap <= 0 or _last_groq_request_done <= 0:
        return
    now = time.time()
    wait = gap - (now - _last_groq_request_done)
    if wait > 0.05:
        print(
            f"[groq] ritmo anti-429: {wait:.1f}s (GROQ_MIN_SECONDS_BETWEEN_REQUESTS={gap})…",
            flush=True,
        )
        time.sleep(wait)


def _groq_mark_request_done() -> None:
    global _last_groq_request_done
    _last_groq_request_done = time.time()


def groq_api_url() -> str:
    return os.environ.get(
        "GROQ_API_URL",
        "https://api.groq.com/openai/v1/chat/completions",
    ).strip()


def groq_model_id() -> str:
    return os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant").strip() or "llama-3.1-8b-instant"


def groq_timeout_s() -> float:
    return float(os.environ.get("GROQ_HTTP_TIMEOUT", "300"))


def _groq_user_agent() -> str:
    # Cloudflare 1010 suele bloquear el User-Agent por defecto de urllib ("Python-urllib/…").
    u = os.environ.get("GROQ_USER_AGENT", "").strip()
    if u:
        return u
    return (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )


def _api_key() -> str:
    k = os.environ.get("GROQ_API_KEY", "").strip()
    if not k:
        raise RuntimeError("Definí GROQ_API_KEY (https://console.groq.com) o usá LLM_PROVIDER=ollama")
    return k


def require_groq_key() -> None:
    _api_key()


def _ssl_context_for_groq() -> ssl.SSLContext:
    """
    Igual criterio que Gemini: macOS / algunos Python fallan con CERTIFICATE_VERIFY_FAILED.
    Preferimos certifi; opcional: GROQ_SSL_CA_BUNDLE o GROQ_INSECURE_SSL=1 (solo desarrollo).
    """
    if os.environ.get("GROQ_INSECURE_SSL", "").lower() in ("1", "true", "yes"):
        return ssl._create_unverified_context()
    bundle = os.environ.get("GROQ_SSL_CA_BUNDLE", "").strip()
    if bundle and os.path.isfile(bundle):
        return ssl.create_default_context(cafile=bundle)
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _strip_ap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_ap(v) for k, v in obj.items() if k != "additionalProperties"}
    if isinstance(obj, list):
        return [_strip_ap(x) for x in obj]
    return obj


def bedrock_tool_schemas_to_openai_tools(schemas: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in schemas:
        params = _strip_ap(s.get("parameters") or {"type": "object", "properties": {}})
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


def bedrock_messages_to_openai(messages: list[dict]) -> list[dict]:
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
                    tid = (tr.get("toolUseId") or "").strip() or str(uuid.uuid4())
                    payload: Any = {}
                    for b in tr.get("content") or []:
                        if "json" in b:
                            payload = b["json"]
                            break
                    body = json.dumps(payload, ensure_ascii=False)
                    omsg.append({"role": "tool", "tool_call_id": tid, "content": body})
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
                                "arguments": json.dumps(inp, ensure_ascii=False),
                            },
                        }
                    )
            assistant: dict[str, Any] = {"role": "assistant"}
            if tool_calls:
                assistant["tool_calls"] = tool_calls
                assistant["content"] = "\n".join(text_bits) if text_bits else None
            else:
                assistant["content"] = "\n".join(text_bits) if text_bits else ""
            omsg.append(assistant)
    return omsg


def _groq_429_wait_seconds(err_body: str) -> float:
    """Parsea 'Please try again in 11.71s' del JSON de error de Groq."""
    m = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)\s*s", err_body, re.IGNORECASE)
    if m:
        return max(1.0, float(m.group(1)) + 0.75)
    return float(os.environ.get("GROQ_429_DEFAULT_WAIT", "15"))


def _groq_429_is_daily_token_limit(err_body: str) -> bool:
    """Cuota diaria (TPD): reintentar en bucle solo quema más tokens / tiempo."""
    low = err_body.lower()
    return "tokens per day" in low or "token limit per day" in low


def _post(payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    ctx = _ssl_context_for_groq()
    timeout = groq_timeout_s()
    max_retries = max(0, int(os.environ.get("GROQ_429_MAX_RETRIES", "12")))
    max_sleep = float(os.environ.get("GROQ_429_MAX_SLEEP", "120"))

    attempt = 0
    had_429_in_this_call = False
    while True:
        req = Request(
            groq_api_url(),
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
                "User-Agent": _groq_user_agent(),
                "Authorization": f"Bearer {_api_key()}",
            },
            method="POST",
        )
        try:
            _groq_pace_before_request()
            with urlopen(req, timeout=timeout, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                _groq_mark_request_done()
                if had_429_in_this_call:
                    chill = float(os.environ.get("GROQ_CHILL_AFTER_429_SEC", "8"))
                    if chill > 0:
                        print(
                            f"[groq] 429 superado: enfriamiento {chill:.0f}s antes del próximo turno "
                            f"(GROQ_CHILL_AFTER_429_SEC; reduce otro 429 seguido)…",
                            flush=True,
                        )
                        time.sleep(chill)
                        _groq_mark_request_done()
                return data
        except HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            _groq_mark_request_done()
            too_big = "Request too large" in err and "reduce your message" in err.lower()
            if e.code == 429 and _groq_429_is_daily_token_limit(err):
                hint = (
                    " Límite **diario** de tokens Groq (TPD) agotado: no reintentamos en bucle. "
                    "Esperá el reset (el mensaje indica minutos), usá otra API key/org, "
                    "**LLM_PROVIDER=ollama** o Bedrock."
                )
                raise RuntimeError(f"Groq HTTP 429 (TPD): {err}{hint}") from e
            if e.code == 429 and attempt < max_retries and not too_big:
                had_429_in_this_call = True
                wait = min(_groq_429_wait_seconds(err), max_sleep)
                # El contador es por **esta** petición HTTP; en cada [paso N] del agente vuelve a 1.
                print(
                    f"[groq] 429 TPM/RPM: esperando {wait:.1f}s "
                    f"(reintento HTTP {attempt + 1}/{max_retries} de esta misma llamada; "
                    f"no es el paso del agente — eso sigue en [paso …] cuando termine)…",
                    flush=True,
                )
                time.sleep(wait)
                attempt += 1
                continue
            hint = ""
            if e.code == 403 and "1010" in err:
                hint = (
                    " (Cloudflare 1010: a veces IP de datacenter/VPN o fingerprint de cliente; "
                    "probá desde red residencial sin VPN, o GROQ_USER_AGENT con otro valor; "
                    "ver https://community.groq.com )"
                )
            elif e.code == 413 or (e.code == 429 and too_big):
                hint = (
                    " Pedido demasiado grande para el tier (~6000 TPM por request con system+tools). "
                    "Bajá GROQ_MAX_CONTEXT_TOKENS_BEFORE_COMPRESS (ej. 2200), GROQ_TOOL_RESULT_MAX_CHARS, "
                    "o MAX_COMPRESS_INPUT_CHARS para el resumen."
                )
            elif e.code == 429:
                hint = (
                    " Aumentá GROQ_429_MAX_RETRIES o GROQ_429_MAX_SLEEP; "
                    "el plan free tiene TPM bajo con historial largo — "
                    "activá compresión de contexto o probá otro GROQ_MODEL en console.groq.com."
                )
            raise RuntimeError(f"Groq HTTP {e.code}: {err}{hint}") from e
        except URLError as e:
            _groq_mark_request_done()
            hint = ""
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                hint = (
                    " Probá: pip install certifi; o en .env GROQ_SSL_CA_BUNDLE=/ruta/ca.pem; "
                    "solo local: GROQ_INSECURE_SSL=1"
                )
            raise RuntimeError(f"Groq red: {e}{hint}") from e


def _groq_is_tool_use_failed(err_msg: str) -> bool:
    return "HTTP 400" in err_msg and "tool_use_failed" in err_msg


def llm_groq(
    _client: Any,
    messages: list[dict],
    system: str,
    tools: Optional[list[dict]] = None,
) -> GeminiAgentResponse:
    tool_retries = max(0, int(os.environ.get("GROQ_TOOL_FAIL_RETRIES", "3")))
    system_base = (system or "").strip()
    if tools and os.environ.get("GROQ_TOOL_FORMAT_HINT", "1").lower() not in ("0", "false", "no"):
        system_base = (system_base + _GROQ_TOOL_FORMAT_RULES).strip()

    last_exc: Optional[Exception] = None
    for attempt in range(tool_retries + 1):
        sys_content = system_base
        if attempt > 0:
            sys_content = (sys_content + "\n\n" + _GROQ_TOOL_FAIL_RETRY_NUDGE).strip()

        msgs: list[dict[str, Any]] = []
        if sys_content:
            msgs.append({"role": "system", "content": sys_content})
        msgs.extend(bedrock_messages_to_openai(messages))

        payload: dict[str, Any] = {
            "model": groq_model_id(),
            "messages": msgs,
            "temperature": float(os.environ.get("GROQ_TEMPERATURE", "0.2")),
            "stream": False,
        }
        if tools:
            payload["tools"] = bedrock_tool_schemas_to_openai_tools(tools)
            payload["tool_choice"] = "auto"
            if os.environ.get("GROQ_PARALLEL_TOOL_CALLS", "1").lower() not in ("0", "false", "no"):
                payload["parallel_tool_calls"] = False

        try:
            raw = _post(payload)
        except RuntimeError as e:
            last_exc = e
            if (
                tools
                and attempt < tool_retries
                and _groq_is_tool_use_failed(str(e))
            ):
                cooldown = float(os.environ.get("GROQ_TOOL_FAIL_COOLDOWN", "10"))
                if cooldown > 0:
                    print(
                        f"[groq] pausa {cooldown:.0f}s tras tool_use_failed (TPM/formato)…",
                        flush=True,
                    )
                    time.sleep(cooldown)
                print(
                    f"[groq] tool_use_failed (intento {attempt + 1}/{tool_retries + 1}), "
                    "reintentando con refuerzo de formato…",
                    flush=True,
                )
                continue
            raise

        last_exc = None
        break

    if last_exc is not None:
        if _groq_is_tool_use_failed(str(last_exc)):
            raise RuntimeError(
                f"{last_exc} "
                "→ Modelos 8B a veces emiten <function=…>; probá "
                "`GROQ_MODEL=llama-3.3-70b-versatile` o subí `GROQ_TOOL_FAIL_COOLDOWN`."
            ) from last_exc
        raise last_exc
    choice = (raw.get("choices") or [{}])[0]
    gmsg = choice.get("message") or {}
    content = gmsg.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)
    tool_calls = gmsg.get("tool_calls") or []

    bd_parts: list[dict[str, Any]] = []
    out: list[Any] = []
    if content.strip():
        bd_parts.append({"text": content})
        out.append(_TextPart(content))
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name") or "unknown"
        args_raw = fn.get("arguments")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError:
                args = {"raw": args_raw}
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {}
        cid = str(tc.get("id") or uuid.uuid4())
        bd_parts.append({"toolUse": {"toolUseId": cid, "name": name, "input": args}})
        out.append(_ToolUsePart(name=name, arguments=json.dumps(args), call_id=cid))

    if not bd_parts:
        bd_parts.append({"text": "(sin texto ni herramientas en la respuesta de Groq)"})
        out.append(_TextPart(bd_parts[0]["text"]))

    bedrock_msg = {"role": "assistant", "content": bd_parts}
    return GeminiAgentResponse(bedrock_msg, out)


def groq_text_only(_client: Any, user_text: str, system: str) -> str:
    msgs: list[dict[str, Any]] = []
    if system and system.strip():
        msgs.append({"role": "system", "content": system.strip()})
    msgs.append({"role": "user", "content": user_text})
    payload = {
        "model": groq_model_id(),
        "messages": msgs,
        "temperature": float(os.environ.get("GROQ_TEMPERATURE", "0.2")),
        "stream": False,
    }
    raw = _post(payload)
    choice = (raw.get("choices") or [{}])[0]
    gmsg = choice.get("message") or {}
    c = gmsg.get("content")
    return (c if isinstance(c, str) else str(c or "")).strip()
