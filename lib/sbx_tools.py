# -*- coding: utf-8 -*-
"""Herramientas de sistema de archivos sobre el sandbox E2B (vía run_code)."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from e2b_code_interpreter import Sandbox


class ToolError(Exception):
    """Error al ejecutar una herramienta en el sandbox (ruta inválida, IO, etc.)."""


# E2B ``run_code`` no garantiza cwd; rutas relativas como ``web-app`` dependen del cwd.
# La plantilla code-interpreter expone ``/code`` como workspace en el explorador; el home
# suele mostrarse aparte. Probamos ``/code`` primero si existe para que ``web-app`` quede
# en ``/code/web-app`` y sea visible donde el usuario mira en el dashboard.
_SANDBOX_CHDIR_HOME = """
import os as __sbx_os
for __p in ("/code", __sbx_os.path.expanduser("~"), "/home/user"):
    if __p and __sbx_os.path.isdir(__p):
        try:
            __sbx_os.chdir(__p)
        except OSError:
            continue
        else:
            break
""".strip()


def with_home_as_cwd(code: str) -> str:
    """Anteponer chdir a /code (si existe), ~ o /home/user para rutas relativas al workspace."""
    return _SANDBOX_CHDIR_HOME + "\n" + code


E2B_TRANSPORT_HINT = (
    "E2B: timeout o red cortada hacia el sandbox (p. ej. Errno 60). "
    "Los archivos del proyecto (p. ej. web-app/src/app/page.tsx) están solo en la VM remota, "
    "no en tu carpeta del repo. Reintentá; para copiarlos al Mac: SANDBOX_EXPORT_DIR=./e2b-export "
    "antes de correr el script. Sandboxes LIVE: e2b.dev → Sandboxes."
)


def is_e2b_transport_error(exc: BaseException) -> bool:
    """Timeouts / cortes de red hacia la API de E2B (p. ej. Errno 60 Operation timed out)."""
    mod = type(exc).__module__ or ""
    name = type(exc).__name__
    if not mod.startswith("httpx"):
        return False
    return name in (
        "ReadError",
        "WriteError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "TimeoutException",
        "RemoteProtocolError",
    )


def _run_py(sbx: "Sandbox", code: str) -> tuple[str, list[str]]:
    try:
        execution = sbx.run_code(with_home_as_cwd(code))
    except BaseException as e:
        if is_e2b_transport_error(e):
            raise ToolError(E2B_TRANSPORT_HINT) from e
        raise
    out = "".join(execution.logs.stdout or [])
    err_lines = list(execution.logs.stderr or [])
    if execution.error:
        err_lines.append(f"{execution.error.name}: {execution.error.value}")
    return out, err_lines


def _parse_json_line(stdout: str) -> Any:
    text = stdout.strip()
    if not text:
        raise ToolError("Salida vacía del sandbox")
    last = text.splitlines()[-1]
    try:
        return json.loads(last)
    except json.JSONDecodeError as e:
        raise ToolError(f"Respuesta no JSON del sandbox: {last[:200]}… ({e})") from e


def list_directory(sbx: "Sandbox", path: str) -> list[str]:
    """Lista archivos y carpetas en `path` (nombres relativos, ordenados)."""
    code = f"""
import json, os
p = {json.dumps(path)}
try:
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    if not os.path.isdir(p):
        raise NotADirectoryError(p)
    names = sorted(os.listdir(p))
    print(json.dumps(names))
except Exception as e:
    print(json.dumps({{"__error__": str(e)}}))
"""
    out, errs = _run_py(sbx, code)
    if errs:
        raise ToolError("; ".join(errs))
    data = _parse_json_line(out)
    if isinstance(data, dict) and "__error__" in data:
        raise ToolError(data["__error__"])
    if not isinstance(data, list):
        raise ToolError(f"list_directory: tipo inesperado {type(data)}")
    return data


def read_file(sbx: "Sandbox", path: str, limit: int | None = None, offset: int = 0) -> str:
    """Lee un archivo de texto UTF-8."""
    lim_py = "None" if limit is None else str(int(limit))
    code = f"""
import json
p = {json.dumps(path)}
off = {int(offset)}
lim = {lim_py}
try:
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        if off:
            f.seek(off)
        if lim is None:
            content = f.read()
        else:
            content = f.read(lim)
    print(json.dumps(content))
except Exception as e:
    print(json.dumps({{"__error__": str(e)}}))
"""
    out, errs = _run_py(sbx, code)
    if errs:
        raise ToolError("; ".join(errs))
    data = _parse_json_line(out)
    if isinstance(data, dict) and "__error__" in data:
        raise ToolError(data["__error__"])
    if not isinstance(data, str):
        raise ToolError("read_file: contenido no es texto")
    return data


def write_file(sbx: "Sandbox", path: str, content: str) -> dict[str, Any]:
    """Escribe un archivo (crea directorios padre si hace falta)."""
    code = f"""
import json, os
p = {json.dumps(path)}
raw = {json.dumps(content)}
try:
    parent = os.path.dirname(p) or "."
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(raw)
    print(json.dumps({{"ok": True, "bytes": len(raw.encode("utf-8")), "path": p}}))
except Exception as e:
    print(json.dumps({{"__error__": str(e)}}))
"""
    out, errs = _run_py(sbx, code)
    if errs:
        raise ToolError("; ".join(errs))
    data = _parse_json_line(out)
    if isinstance(data, dict) and "__error__" in data:
        raise ToolError(data["__error__"])
    return data


def search_file_content(
    sbx: "Sandbox",
    pattern: str,
    max_results: int = 50,
    root: str = ".",
) -> dict[str, Any]:
    """
    Busca `pattern` como expresión regular en archivos de texto bajo `root`.
    Devuelve JSON con matches paginados y bandera truncated.
    """
    code = f"""
import json, os, re
pat = re.compile({json.dumps(pattern)})
root = {json.dumps(root)}
max_r = {int(max_results)}
matches = []
truncated = False
skip_dirs = {{".git", "node_modules", ".next", "__pycache__", ".venv", "venv"}}

def is_text(path):
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
        if b"\\x00" in chunk:
            return False
        chunk.decode("utf-8")
        return True
    except Exception:
        return False

for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in skip_dirs]
    for name in filenames:
        if len(matches) >= max_r:
            truncated = True
            break
        fp = os.path.join(dirpath, name)
        if not is_text(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if len(matches) >= max_r:
                        truncated = True
                        break
                    if pat.search(line):
                        matches.append({{"path": fp, "line": i, "text": line.rstrip()[:500]}})
                if truncated:
                    break
        except Exception:
            continue
    if truncated:
        break
print(json.dumps({{"matches": matches, "truncated": truncated, "count": len(matches)}}))
"""
    out, errs = _run_py(sbx, code)
    if errs:
        raise ToolError("; ".join(errs))
    data = _parse_json_line(out)
    if not isinstance(data, dict):
        raise ToolError("search_file_content: formato inválido")
    return data


def replace_in_file(sbx: "Sandbox", path: str, old: str, new: str) -> dict[str, Any]:
    """Reemplaza la primera ocurrencia de `old` por `new` en el archivo."""
    code = f"""
import json
p = {json.dumps(path)}
old_s = {json.dumps(old)}
new_s = {json.dumps(new)}
try:
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()
    if old_s not in text:
        print(json.dumps({{"ok": False, "reason": "old not found", "path": p}}))
    else:
        text2 = text.replace(old_s, new_s, 1)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text2)
        print(json.dumps({{"ok": True, "path": p, "replaced_once": True}}))
except Exception as e:
    print(json.dumps({{"__error__": str(e)}}))
"""
    out, errs = _run_py(sbx, code)
    if errs:
        raise ToolError("; ".join(errs))
    data = _parse_json_line(out)
    if isinstance(data, dict) and "__error__" in data:
        raise ToolError(data["__error__"])
    return data


def glob(sbx: "Sandbox", pattern: str, root: str = ".") -> list[str]:
    """Busca rutas con patrón estilo glob (ej. `**/*.tsx`), desde `root`."""
    code = f"""
import json
from pathlib import Path
base = Path({json.dumps(root)})
pat = {json.dumps(pattern)}
try:
    paths = sorted(str(p) for p in base.glob(pat) if p.is_file())
    print(json.dumps(paths))
except Exception as e:
    print(json.dumps({{"__error__": str(e)}}))
"""
    out, errs = _run_py(sbx, code)
    if errs:
        raise ToolError("; ".join(errs))
    data = _parse_json_line(out)
    if isinstance(data, dict) and "__error__" in data:
        raise ToolError(data["__error__"])
    if not isinstance(data, list):
        raise ToolError("glob: resultado inválido")
    return data
