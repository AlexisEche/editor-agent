# -*- coding: utf-8 -*-
"""Google Gemini (REST v1beta) con historial compatible con el formato Bedrock del agente."""

from __future__ import annotations

import json
import os
import re
import socket
import ssl
import time
import uuid
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _ssl_context_for_gemini() -> ssl.SSLContext:
    """
    macOS / algunos Python suelen fallar con CERTIFICATE_VERIFY_FAILED al usar urllib
    sin bundle explícito. Preferimos certifi; opcional: GEMINI_SSL_CA_BUNDLE o
    GEMINI_INSECURE_SSL=1 (solo desarrollo local).
    """
    if os.environ.get("GEMINI_INSECURE_SSL", "").lower() in ("1", "true", "yes"):
        return ssl._create_unverified_context()
    bundle = os.environ.get("GEMINI_SSL_CA_BUNDLE", "").strip()
    if bundle and os.path.isfile(bundle):
        return ssl.create_default_context(cafile=bundle)
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


# Sin gemini-flash-latest (Gemini 3 + historial de tools de 2.x → 400 thought_signature).
# Cadena: cuota **por modelo** en free tier. **Sin** gemini-2.5-pro por defecto (suele venir con limit:0 sin facturación).
# Con billing en Google AI: añadí ",gemini-2.5-pro" en GEMINI_MODEL.
_DEFAULT_MODEL_CHAIN = (
    "gemini-2.5-flash-lite,gemini-2.5-flash,gemini-2.0-flash-lite,gemini-2.0-flash"
)

# Tras el primer generateContent exitoso, se reintenta ese modelo primero (menos 429 en cadena).
_sticky_success_model: Optional[str] = None


def gemini_model_chain() -> list[str]:
    """Lista de modelos: GEMINI_MODEL separado por coma; fallback ante 429, 404 y 400 (thought_signature)."""
    raw = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL_CHAIN).strip()
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return parts if parts else ["gemini-2.5-flash-lite"]


def gemini_model_id() -> str:
    """Primer modelo de la cadena (compatibilidad)."""
    return gemini_model_chain()[0]


def _ordered_model_chain() -> list[str]:
    """Cadena GEMINI_MODEL; si hubo un modelo que ya respondió OK, probarlo primero."""
    chain = list(gemini_model_chain())
    if os.environ.get("GEMINI_NO_STICKY_MODEL", "").lower() in ("1", "true", "yes"):
        return chain
    sticky = _sticky_success_model
    if sticky and sticky in chain:
        return [sticky] + [m for m in chain if m != sticky]
    return chain


def reset_gemini_sticky_model() -> None:
    """Tests o nuevo agente: olvidar el modelo que venía funcionando."""
    global _sticky_success_model
    _sticky_success_model = None


def _format_gemini_429(err_body: str) -> str:
    """Explica 429 / RESOURCE_EXHAUSTED (cuota free tier = 0, RPM, etc.)."""
    low = err_body.lower()
    limit_zero_ft = "limit: 0" in low and "free_tier" in low
    limit_zero_note = ""
    if limit_zero_ft:
        limit_zero_note = (
            "\n[gemini] Si ves **limit: 0** en `generate_content_free_tier_*`: para ese modelo/proyecto Google **no hay cupo "
            "gratis asignado** (o está anulado). No se arregla cambiando código: hace falta **facturación** en el proyecto "
            "de la API key (Google AI Studio → proyecto → Billing), **otra clave/proyecto** con cuota, o **LLM_PROVIDER=bedrock** / **ollama**.\n"
        )
    hint = (
        limit_zero_note
        + "\n\n[gemini] Qué podés hacer:\n"
        "  • **Uso inmediato:** en `.env` poné **LLM_PROVIDER=bedrock** y credenciales AWS (ya soportado por `tarea/agent_web_dev.py`) "
        "si Gemini agotó la cuota en todos los modelos de la lista.\n"
        "  • **limit: 0** en free tier: a veces el modelo **no tiene cupo** en ese proyecto, o hace falta **facturación** en Google AI / Cloud "
        "para recuperar cuota — revisá https://ai.google.dev/gemini-api/docs/rate-limits\n"
        "  • Variá modelos: **GEMINI_MODEL** (coma). **gemini-2.5-pro** solo si tenés facturación/cuota (sino limit:0).\n"
        "  • Si **todos** los Flash dan 429 seguidos: cupo **RPM/RPD** del proyecto agotado — esperá o habilitá billing / otro proyecto.\n"
        "  • **GEMINI_429_MAX_DELAY** (default 12): tope de segundos de espera ante 429 (evita ~58s × N por cada paso).\n"
        "  • **GEMINI_429_PER_MODEL_RETRIES** (default 2): reintentos por modelo.\n"
        "  • Con **limit: 0** en el cuerpo del 429, por defecto **no** se espera: se pasa al siguiente modelo (**GEMINI_429_SKIP_WAIT_ON_LIMIT_ZERO**).\n"
        "  • **LLM_PROVIDER=ollama** para modelo local sin cuota de Google.\n"
    )
    return f"Gemini HTTP 429 (cuota / límite de uso):\n{err_body}{hint}"


def _api_key() -> str:
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if not k:
        raise RuntimeError("Definí GEMINI_API_KEY en el entorno o en .env")
    return k


def _clean_schema(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _clean_schema(v)
            for k, v in obj.items()
            if k != "additionalProperties"
        }
    if isinstance(obj, list):
        return [_clean_schema(x) for x in obj]
    return obj


_JSON_TYPES = frozenset({"object", "string", "integer", "number", "boolean", "array", "null"})


def _uppercase_schema_types(obj: Any) -> Any:
    """Gemini REST suele esperar tipos JSON Schema en mayúsculas (OBJECT, STRING, …)."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "type" and isinstance(v, str) and v.lower() in _JSON_TYPES:
                out[k] = v.upper()
            else:
                out[k] = _uppercase_schema_types(v)
        return out
    if isinstance(obj, list):
        return [_uppercase_schema_types(x) for x in obj]
    return obj


def _thought_signature_from_part(part: dict) -> Any:
    """
    Firma que Gemini 3 devuelve en cada part (text/functionCall); hay que reenviarla igual.
    None = la API no mandó la clave (no incluir al reconstruir el request).
    """
    if "thoughtSignature" in part:
        return part["thoughtSignature"]
    if "thought_signature" in part:
        return part["thought_signature"]
    return None


def schemas_to_gemini_tools(schemas: list[dict]) -> list[dict]:
    decls: list[dict] = []
    for s in schemas:
        raw = _clean_schema(s.get("parameters") or {"type": "object", "properties": {}})
        params = _uppercase_schema_types(raw)
        decls.append(
            {
                "name": s["name"],
                "description": s.get("description", ""),
                "parameters": params,
            }
        )
    return [{"functionDeclarations": decls}]


def _bedrock_user_parts_to_gemini(parts: list[dict]) -> list[dict]:
    gp: list[dict] = []
    for p in parts:
        if "text" in p:
            gp.append({"text": p["text"]})
        elif "toolResult" in p:
            tr = p["toolResult"]
            name = tr.get("name")
            if not name:
                name = "unknown_tool"
            payload: Any = {}
            for block in tr.get("content") or []:
                if "json" in block:
                    payload = block["json"]
                    break
            gp.append({"functionResponse": {"name": name, "response": {"result": payload}}})
    return gp


def _bedrock_assistant_parts_to_gemini(parts: list[dict]) -> list[dict]:
    gp: list[dict] = []
    for p in parts:
        sig = _thought_signature_from_part(p)
        if "text" in p and p.get("text"):
            block: dict[str, Any] = {"text": p["text"]}
            if sig is not None:
                block["thoughtSignature"] = sig
            gp.append(block)
        elif "toolUse" in p:
            tu = p["toolUse"]
            args = tu.get("input")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            elif args is None:
                args = {}
            fc: dict[str, Any] = {"name": tu["name"], "args": args}
            if tu.get("toolUseId"):
                fc["id"] = tu["toolUseId"]
            block = {"functionCall": fc}
            if sig is not None:
                block["thoughtSignature"] = sig
            gp.append(block)
    return gp


def bedrock_messages_to_gemini_contents(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        parts = msg.get("content") or []
        if role == "user":
            gp = _bedrock_user_parts_to_gemini(parts)
            if gp:
                out.append({"role": "user", "parts": gp})
        elif role == "assistant":
            gp = _bedrock_assistant_parts_to_gemini(parts)
            if gp:
                out.append({"role": "model", "parts": gp})
    return out


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


class GeminiAgentResponse:
    def __init__(self, bedrock_message: dict, output: list) -> None:
        self._bedrock_message = bedrock_message
        self.output = output

    def bedrock_message(self) -> dict:
        return self._bedrock_message


class _GeminiHTTP429(Exception):
    __slots__ = ("body",)

    def __init__(self, body: str) -> None:
        self.body = body


class _GeminiHTTPRetryNextModel(Exception):
    """404, 400 (thought_signature), 502/503 (saturation): probar siguiente modelo de la cadena."""

    __slots__ = ("code", "body")

    def __init__(self, code: int, body: str) -> None:
        self.code = code
        self.body = body


def _parse_retry_delay_seconds(err_body: str) -> float:
    """Extrae segundos sugeridos por la API (RetryInfo o texto 'retry in Xs')."""
    try:
        data = json.loads(err_body)
        details = (data.get("error") or {}).get("details") or []
        for d in details:
            if isinstance(d, dict) and "RetryInfo" in str(d.get("@type", "")):
                rd = d.get("retryDelay")
                if isinstance(rd, str) and rd.endswith("s"):
                    return min(120.0, float(rd[:-1]) + 0.5)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    m = re.search(r"retry in ([0-9.]+)\s*s", err_body, re.I)
    if m:
        return min(120.0, float(m.group(1)) + 0.5)
    return float(os.environ.get("GEMINI_429_FALLBACK_DELAY", "6"))


def _gemini_429_body_suggests_limit_zero(body: str) -> bool:
    """Si el mensaje incluye limit:0, reintentar 58s no suele desbloquear cupo: mejor otro modelo."""
    return "limit: 0" in body.lower() or "limit\": 0" in body.lower()


def _gemini_http_timeout_s() -> float:
    """Timeout total de lectura HTTP (generateContent puede tardar mucho con tools + historial largo)."""
    return float(os.environ.get("GEMINI_HTTP_TIMEOUT", "600"))


def _effective_429_wait_seconds(err_body: str) -> float:
    """Espera sugerida por Google acotada (evita bloqueos de ~1 min × muchos reintentos)."""
    raw = _parse_retry_delay_seconds(err_body)
    cap = float(os.environ.get("GEMINI_429_MAX_DELAY", "12"))
    floor = float(os.environ.get("GEMINI_429_MIN_DELAY", "2"))
    return max(floor, min(raw, cap))


def _is_timeout_exc(exc: BaseException) -> bool:
    if isinstance(exc, socket.timeout):
        return True
    if type(exc).__name__ == "TimeoutError":
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (110, 60):  # ETIMEDOUT varios SO
        return True
    if isinstance(exc, URLError) and isinstance(getattr(exc, "reason", None), socket.timeout):
        return True
    return False


def _post_generate_once(model_id: str, body: dict) -> dict:
    key = _api_key()
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={key}"
    )
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    ctx = _ssl_context_for_gemini()
    timeout = _gemini_http_timeout_s()
    timeout_retries = max(1, int(os.environ.get("GEMINI_HTTP_TIMEOUT_RETRIES", "2")))

    for t_attempt in range(timeout_retries):
        try:
            with urlopen(req, timeout=timeout, context=ctx) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code == 429:
                raise _GeminiHTTP429(err_body) from e
            if e.code in (502, 503):
                raise _GeminiHTTPRetryNextModel(e.code, err_body) from e
            if e.code == 404:
                raise _GeminiHTTPRetryNextModel(404, err_body) from e
            if e.code == 400:
                low = err_body.lower()
                if "thought_signature" in low or "thoughtsignature" in low:
                    raise _GeminiHTTPRetryNextModel(400, err_body) from e
            raise RuntimeError(f"Gemini HTTP {e.code} (modelo `{model_id}`): {err_body}") from e
        except URLError as e:
            if _is_timeout_exc(e):
                if t_attempt + 1 < timeout_retries:
                    print(
                        f"[gemini] timeout leyendo respuesta ({timeout}s) — reintento "
                        f"{t_attempt + 2}/{timeout_retries}…",
                        flush=True,
                    )
                    time.sleep(2.0)
                    continue
                raise RuntimeError(
                    f"Gemini: timeout tras {timeout_retries} intentos (GEMINI_HTTP_TIMEOUT={timeout}s). "
                    "Subí GEMINI_HTTP_TIMEOUT en .env, o activá DISABLE_CONTEXT_COMPRESSION=true para acortar el payload."
                ) from e
            raise RuntimeError(f"Gemini red: {e}") from e
        except (socket.timeout, TimeoutError) as e:
            if t_attempt + 1 < timeout_retries:
                print(
                    f"[gemini] socket timeout ({timeout}s) — reintento {t_attempt + 2}/{timeout_retries}…",
                    flush=True,
                )
                time.sleep(2.0)
                continue
            raise RuntimeError(
                f"Gemini: la API no respondió a tiempo ({timeout}s × {timeout_retries}). "
                "Aumentá GEMINI_HTTP_TIMEOUT (p. ej. 900) en .env."
            ) from e


def _post_generate_one_model_with_429_retries(model_id: str, body: dict) -> dict:
    """
    Ante 429, reintenta el mismo modelo con espera acotada (GEMINI_429_MAX_DELAY).
    Si el cuerpo indica limit:0, no espera: se deja que la cadena pruebe otro modelo.
    """
    retries = max(1, int(os.environ.get("GEMINI_429_PER_MODEL_RETRIES", "2")))
    skip_wait = os.environ.get("GEMINI_429_SKIP_WAIT_ON_LIMIT_ZERO", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    for attempt in range(retries):
        try:
            return _post_generate_once(model_id, body)
        except _GeminiHTTP429 as e:
            if skip_wait and _gemini_429_body_suggests_limit_zero(e.body):
                print(
                    f"[gemini] 429 en `{model_id}` (cuota/limit:0 en respuesta — sin espera larga, "
                    "siguiente modelo de la lista)",
                    flush=True,
                )
                raise
            if attempt + 1 >= retries:
                raise
            delay = _effective_429_wait_seconds(e.body)
            print(
                f"[gemini] 429 en `{model_id}` (reintento {attempt + 1}/{retries}), "
                f"esperando {delay:.1f}s (tope GEMINI_429_MAX_DELAY)…",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("Gemini: fallo interno en reintentos 429")


def _post_generate(body: dict) -> dict:
    """Prueba cada modelo de la cadena hasta que uno responda (429, 404, 400, 502/503)."""
    global _sticky_success_model
    chain = _ordered_model_chain()
    last_429: Optional[_GeminiHTTP429] = None
    last_retry: Optional[_GeminiHTTPRetryNextModel] = None
    for i, mid in enumerate(chain):
        try:
            out = _post_generate_one_model_with_429_retries(mid, body)
            _sticky_success_model = mid
            return out
        except _GeminiHTTP429 as e:
            last_429 = e
            if _sticky_success_model == mid:
                _sticky_success_model = None
            if i + 1 < len(chain):
                nxt = chain[i + 1]
                print(
                    f"[gemini] 429 en `{mid}` → probando `{nxt}`…",
                    flush=True,
                )
                continue
            raise RuntimeError(_format_gemini_429(e.body)) from e
        except _GeminiHTTPRetryNextModel as e:
            last_retry = e
            if i + 1 < len(chain):
                nxt = chain[i + 1]
                if e.code == 404:
                    reason = "no disponible o no soportado"
                elif e.code == 400:
                    reason = "thought_signature / historial (evitar Gemini 3 tras 2.5 sin firmas)"
                elif e.code in (502, 503):
                    reason = "servicio saturado o no disponible (temporal)"
                else:
                    reason = "reintentar con otro modelo"
                print(
                    f"[gemini] HTTP {e.code} con `{mid}` ({reason}) → probando `{nxt}`…",
                    flush=True,
                )
                continue
            hint = ""
            if e.code == 400 and ("thought_signature" in e.body.lower() or "thoughtsignature" in e.body.lower()):
                hint = (
                    "\n\n[gemini] Este 400 suele ser **Gemini 3** (`gemini-flash-latest`, etc.) con historial de "
                    "herramientas creado por **2.5/2.0** (sin `thoughtSignature`). "
                    "Quitá modelos Gemini 3 de `GEMINI_MODEL` o usá solo uno desde el primer turno, o **Bedrock**.\n"
                )
            raise RuntimeError(
                f"Gemini HTTP {e.code} (modelo `{mid}`): {e.body}{hint}"
            ) from e
    if last_429:
        raise RuntimeError(_format_gemini_429(last_429.body)) from last_429
    if last_retry:
        raise RuntimeError(f"Gemini HTTP {last_retry.code}: {last_retry.body}") from last_retry
    raise RuntimeError("Gemini: GEMINI_MODEL vacío o inválido")


def llm_gemini(
    _client: Any,
    messages: list[dict],
    system: str,
    tools: Optional[list[dict]] = None,
) -> GeminiAgentResponse:
    contents = bedrock_messages_to_gemini_contents(messages)
    body: dict[str, Any] = {
        "contents": contents,
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        # Menos “charla”; más inclinación a usar herramientas cuando el prompt lo pide.
        "generationConfig": {
            "temperature": float(os.environ.get("GEMINI_TEMPERATURE", "0.2")),
        },
    }
    if system and system.strip():
        body["systemInstruction"] = {"parts": [{"text": system.strip()}]}
    if tools:
        body["tools"] = schemas_to_gemini_tools(tools)

    raw = _post_generate(body)
    pf = raw.get("promptFeedback") or {}
    if pf.get("blockReason"):
        raise RuntimeError(f"Gemini bloqueó el prompt: {pf}")
    cands = raw.get("candidates") or []
    if not cands:
        raise RuntimeError(f"Gemini no devolvió candidates: {json.dumps(raw)[:800]}")
    cand = cands[0]
    g_content = cand.get("content") or {}
    g_parts = g_content.get("parts") or []

    bd_parts: list[dict] = []
    out: list[Any] = []

    for part in g_parts:
        tsig = _thought_signature_from_part(part)
        if "functionCall" in part:
            fc = part["functionCall"] or {}
            name = fc.get("name") or "unknown"
            args = fc.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            cid = fc.get("id") or str(uuid.uuid4())
            block: dict[str, Any] = {"toolUse": {"toolUseId": cid, "name": name, "input": args}}
            if tsig is not None:
                block["thoughtSignature"] = tsig
            bd_parts.append(block)
            out.append(_ToolUsePart(name=name, arguments=json.dumps(args), call_id=cid))
        elif "text" in part and part["text"]:
            t = part["text"]
            block = {"text": t}
            if tsig is not None:
                block["thoughtSignature"] = tsig
            bd_parts.append(block)
            out.append(_TextPart(t))

    bedrock_msg = {"role": "assistant", "content": bd_parts}
    return GeminiAgentResponse(bedrock_msg, out)


def gemini_text_only(_client: Any, user_text: str, system: str) -> str:
    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
    }
    if system and system.strip():
        body["systemInstruction"] = {"parts": [{"text": system.strip()}]}
    raw = _post_generate(body)
    pf = raw.get("promptFeedback") or {}
    if pf.get("blockReason"):
        raise RuntimeError(f"Gemini bloqueó el prompt: {pf}")
    cands = raw.get("candidates") or []
    if not cands:
        raise RuntimeError(f"Gemini no devolvió candidates: {json.dumps(raw)[:800]}")
    cand = cands[0]
    parts = (cand.get("content") or {}).get("parts") or []
    texts = [p.get("text", "") for p in parts if "text" in p]
    return "\n".join(texts).strip()
