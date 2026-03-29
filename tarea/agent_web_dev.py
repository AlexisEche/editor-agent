# -*- coding: utf-8 -*-
"""
Tarea final — Agente de código full stack: instrucción en lenguaje natural → app web
completa en sandbox E2B; lectura/escritura de archivos, verificación de build y
**Runtime Summary** (compresión del 70 % más antiguo vía `lib/context_compression`).

LLM: **Ollama** (local, recomendado sin cuota cloud), **Bedrock**, **Gemini**, **Groq**.
Ejecutar desde la raíz del repo:
  export E2B_API_KEY=...
  # LLM_PROVIDER=groq | ollama | gemini | bedrock
  # Opcional: copiar todo el proyecto Next (sin node_modules/.next) a tu máquina antes de cerrar:
  # export SANDBOX_EXPORT_DIR=./e2b-export
  # Luego: ./scripts/run-exported-next.sh ./e2b-export/web-app
  # Opcional: no aplicar parche post-bootstrap (fuentes geist + next.config): SKIP_GEIST_PATCH=1
  # Opcional: matar TODOS los sandboxes LIVE de esta API key antes de arrancar (libera cuota):
  # export E2B_KILL_RUNNING_SANDBOXES=1
  # Opcional: borrar y recrear la carpeta del proyecto dentro del sandbox (inicio limpio de web-app/):
  # export SANDBOX_RESET_WORKDIR=1
  python tarea/agent_web_dev.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Optional, Tuple

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# override=True: el .env manda sobre variables ya exportadas en la shell (evita quedar pegado a Nova).
load_dotenv(_ROOT / ".env", override=True)

import boto3
from botocore.exceptions import ClientError
from e2b_code_interpreter import Sandbox

from lib.bedrock_llm import bedrock_model_id, converse_text_only, llm
from lib.context_compression import maybe_compress
from lib import gemini_llm
from lib import groq_llm
from lib import ollama_llm
from lib import sbx_tools

# Carpeta creada al iniciar el sandbox (configurable). El agente debe generar Next.js aquí dentro.
def _default_work_dir() -> str:
    return os.environ.get("SANDBOX_PROJECT_DIR", "web-app").strip() or "web-app"


def _groq_compress_token_limit() -> int:
    """Historial (solo mensajes): tope bajo porque Groq free cuenta system+tools+msgs ~6000 TPM por request."""
    raw = os.environ.get("GROQ_MAX_CONTEXT_TOKENS_BEFORE_COMPRESS", "2600").strip()
    try:
        return max(400, int(raw))
    except ValueError:
        return 2600


def _groq_tool_result_max_chars() -> int:
    raw = os.environ.get("GROQ_TOOL_RESULT_MAX_CHARS", "8000").strip()
    try:
        return max(800, int(raw))
    except ValueError:
        return 8000


def _ollama_compress_token_limit() -> int:
    """Modelos locales suelen tener contexto menor que 40k; comprimir antes evita timeouts y errores."""
    raw = os.environ.get("OLLAMA_MAX_CONTEXT_TOKENS_BEFORE_COMPRESS", "8000").strip()
    try:
        return max(1200, int(raw))
    except ValueError:
        return 8000


def _ollama_tool_result_max_chars() -> int:
    raw = os.environ.get("OLLAMA_TOOL_RESULT_MAX_CHARS", "12000").strip()
    try:
        return max(2000, int(raw))
    except ValueError:
        return 12000


def _print_groq_startup(model: str) -> None:
    print(f"[groq] modelo `{model}`", flush=True)
    print(
        "[groq] Tip free tier: 5s entre llamadas por defecto (GROQ_MIN_SECONDS_BETWEEN_REQUESTS); "
        "mucho tool_use_failed → GROQ_MODEL=llama-3.3-70b-versatile",
        flush=True,
    )


def _print_ollama_startup(model: str, base_url: str) -> None:
    print(f"[ollama] {base_url} · modelo `{model}`", flush=True)
    print(
        "[ollama] Si el agente falla en tools o tarda mucho: `OLLAMA_MODEL` más capaz "
        "(p. ej. qwen2.5-coder:7b, llama3.1:8b) y `ollama pull <nombre>`. "
        "RAM insuficiente → modelo más chico o cuantización Q4.",
        flush=True,
    )


def _truncate_tool_result(result: Any, max_chars: int, label: str = "LLM") -> Any:
    try:
        s = json.dumps(result, ensure_ascii=False, default=str)
    except TypeError:
        s = str(result)
    if len(s) <= max_chars:
        return result
    preview = s[: max(0, max_chars - 200)]
    return {
        "_truncated": True,
        "_original_chars": len(s),
        "_preview": preview + "…",
        "_hint": f"Salida truncada ({label}). Leé archivos con read_file o repetí con comando más acotado.",
    }


SYSTEM_PROMPT_WEB_DEV_TEMPLATE = """Sos el **agente de código full stack** de la tarea: tu único objetivo es **construir o modificar una app web completa** en el sandbox. No des charla general, tutoriales abstractos ni respuestas fuera de ese objetivo salvo que el usuario pida explícitamente solo aclaración breve.

El usuario **no** ve el disco: todo lo averiguás con herramientas.

## Workspace fijo (ya existe en el sandbox)
- La carpeta `{work_dir}` **ya fue creada** al iniciar. El runtime intenta dejar **Next.js 14** ya generado ahí (bootstrap con `create-next-app@14.2.18`); verificá con `read_file("{work_dir}/package.json")` **antes** de volver a ejecutar `create-next-app`.
- **No existe** la herramienta `create-next-app` como tal: crear o rehacer el esqueleto Next.js es siempre con **`execute_code`** y `npx`/`subprocess`, o editando archivos en `{work_dir}`.
- **Creá el proyecto Next.js dentro de `{work_dir}`** (no inventes otro nombre salvo que renombres después con herramientas). Ejemplo con `execute_code` (sin TTY: evita el wizard "recommended defaults" de `@latest`):
  - **Mal:** correr `create-next-app` **desde el home** con un argumento posicional `{work_dir}` (p. ej. `"web-app"` al final): eso lo toma como *nombre* del proyecto y dispara el wizard "What is your project named?".
  - **Bien:** `subprocess.run(..., cwd="{work_dir}", env=dict(os.environ, CI="1"), stdin=subprocess.DEVNULL, check=False, timeout=900)` y en la lista de args el destino es **`"."`** (punto), no el nombre de la carpeta.
  - Comando recomendado (versión fija, no `@latest`): `["npx", "create-next-app@14.2.18", "--yes", ".", "--typescript", "--tailwind", "--eslint", "--app", "--src-dir", "--no-import-alias", "--use-npm"]` — el `--yes` va **justo después** del nombre del paquete; **no** al final.
- **npm:** instalá solo paquetes publicados con un solo nombre (ej. `react-icons`, `@heroicons/react`). **No** uses rutas como `@heroicons/react/solid` ni `@heroicons/react/outline` como nombre de paquete en `npm install` (no existen en el registry; dan ENOENT).
- Si `create-next-app` imprime preguntas interactivas o se queda colgado, reintentá con la misma versión **14.2.18** y comprobá que `stdin=subprocess.DEVNULL` y `CI="1"` estén en `subprocess.run`.
- Si `read_file` de `package.json` ya muestra **JSON** con `"next"` en `dependencies` y scripts `dev`/`build`, **no** vuelvas a ejecutar `create-next-app`: editá archivos y corré `npm run build`.
- Con **`--src-dir`** (bootstrap por defecto), la app vive en **`{work_dir}/src/app/page.tsx`** (y `layout.tsx`, `globals.css`). **Prohibido** usar `{work_dir}/page.tsx` en la raíz del proyecto como página principal: Next **no** la usa para `/`. **Prohibido** pegar `<!DOCTYPE html>` o `<html>...</html>` dentro de `.tsx`: es inválido; solo **JSX** en el componente `Page`.
- **Imports:** si el bootstrap usó **`--no-import-alias`**, no asumas `@/…`. Desde **`src/app/page.tsx`** hacia **`src/components/Foo.tsx`** usá **`import Foo from "../components/Foo"`** (no `./components/Foo`, esa carpeta no existe bajo `app/`).
- **Rutas en `write_file`:** usá siempre el prefijo **`{work_dir}/`** (p. ej. `{work_dir}/src/app/page.tsx`). Si mandás solo `src/...`, el runtime lo reubica bajo `{work_dir}/`, pero es mejor ser explícito.
- **Prioridad del trabajo:** lo que importa es **TypeScript/TSX, JS, CSS** y que **`npm run build`** pase. **No** pierdas pasos buscando colecciones de **`.svg` / imágenes** en carpetas inventadas (`components/icons`, etc.) salvo petición explícita del usuario. Nav/íconos “invisibles” → arreglalo con **clases CSS** (`globals.css`, `className`, `fill`/`color` en JSX) o **`next/image`** apuntando a `public/`, no excavando SVGs inexistentes.
- **No** ejecutes en `execute_code` **`npm run dev`**, **`npm start`** ni **`next dev`**: son procesos que **no terminan** y la herramienta hace **timeout**. Para validar el proyecto alcanza **`npm run build`** con `cwd="{work_dir}"`. En el cierre del agente indicá que el usuario puede correr `npm run dev` en su máquina.
- Usá **siempre** `subprocess.run` con **lista** de argumentos (no shell=True) para evitar errores de comillas y paréntesis.

## package.json
- **Solo JSON** (comillas dobles en claves y strings). Nunca YAML (`name: proyecto`). Si `write_file` rechaza el contenido, corregí hasta que `json.loads` sea válido.

## Sandbox **sin pantalla** (E2B = servidor, no escritorio)
- **Prohibido** en `execute_code`: **tkinter**, **PyQt**, **PySide**, **wxPython**, **pygame** con ventanas, o cualquier GUI que necesite `$DISPLAY`. Fallan con `TclError` / `no display`.
- **“Estilo Windows 95”** = **apariencia** en la **web** (bordes grises, fuentes, botones) con **React + CSS Modules o Tailwind** en `{work_dir}/src/app/`, **no** aplicaciones de escritorio.
- Para la app: **read_file** / **write_file** / **replace_in_file** sobre `.tsx` y `.css`, y **subprocess** para `npm run build` con `cwd="{work_dir}"`.

## execute_code (Python) — evitar SyntaxError
- **No** uses `default_api`, APIs de Cursor/IDE ni nada que no exista en el sandbox: solo **stdlib** (`os`, `subprocess`, `json`, `pathlib`, etc.).
- No mezcles comillas sin cerrar. Para strings largos usá literales con comillas triples (docstrings) o variables.
- Antes de `npx`/`npm`, verificá con `list_directory` que `{work_dir}` existe.
- Si recibís error de sintaxis en el resultado de la herramienta, **corregí y reintentá** en el siguiente paso con código más corto y probado.

## Estrategia en dos fases (orden estricto)
1. **Fase mínima (obligatoria primero):** una app **mínima que compile**. Preferí **todo en un solo** `{work_dir}/src/app/page.tsx` con lógica simple (lista estática o pocos `useState`). **No** añadas `src/components/` ni diseño “Windows 95” hasta tener **`npm run build` en verde** al menos una vez.
2. **Fase adicional (después):** recién ahí refinás estilos, extraés componentes, tocás `globals.css`, contraste del nav, etc. **Infraestructura = código fuente + build OK**; assets gráficos son opcionales.

## Anti-bucle (costo en pasos)
- **No repitas** `write_file` con el **mismo** archivo y el **mismo** contenido: el runtime lo rechaza. Si ya escribiste, el siguiente paso es **`execute_code` + `npm run build`** o corregir según el error del build.
- Tras **2–3** ediciones de código, corré **build**; no acumules muchos `write_file` seguidos sin verificar.

## Obligatorio (no negociable)
- **Nunca** pidas rutas ni “¿dónde está X?”. Usá `list_directory`, `glob`, `read_file`, `search_file_content`.
- Ante crear app o corregir UI (nav, íconos, etc.): **en el mismo turno** llamá **al menos una herramienta**. No respondas solo con texto pidiendo datos al usuario.
- **Verificación:** antes de dar por cerrado un bloque de trabajo, comprobá que el proyecto **compila**: ejecutá vía `execute_code` algo equivalente a `npm run build` dentro del directorio del proyecto (y si falla, leé el error, corregí y repetí). Opcional: `npm run lint` si existe.
- No cerrés con solo planes: **archivos creados o modificados** en el sandbox.

## Stack
**Next.js** (App Router), **TypeScript**, **React**; CSS Modules o Tailwind. Estética elaborada = **solo después** de que el build mínimo pase.
- **Fuentes:** el runtime ya dejó **`geist`** + `layout.tsx` con `GeistSans`/`GeistMono` (sin `next/font/local` ni `.woff` en `src/app/fonts/` — en el sandbox suelen romper el build con error de fontkit/`ascent`). Si editás `layout.tsx`, mantené imports desde `geist/font/sans` y `geist/font/mono`, o usá `next/font/google`.

## Herramientas
`list_directory`, `glob`, `read_file`, `write_file`, `search_file_content`, `replace_in_file`, `execute_code` (Python + `subprocess` para npm/node).

## Mismo historial / seguimiento
Si el usuario pide un ajuste después de crear el proyecto: el código **sigue en el mismo sandbox**. No digas que no tenés acceso a archivos: explorá con herramientas y editá.

## Cierre
Solo cuando exista `package.json` y hayas verificado build (o documentado el fallo tras reintentos): **español**, archivos tocados, cómo probar (`npm run dev`) y si el **build** pasó. Basate en lo leído del repo.
"""


def system_prompt_for_workdir(work_dir: str) -> str:
    return SYSTEM_PROMPT_WEB_DEV_TEMPLATE.format(work_dir=work_dir)


SYSTEM_PROMPT_WEB_DEV = system_prompt_for_workdir(_default_work_dir())

# Tras create-next-app, `next/font/local` con .woff en src/app/fonts a veces rompe el build (fontkit / ascent).
# El runtime aplica `_patch_nextjs_geist_stack` en el sandbox; este layout coincide con ese parche.
_NEXT_LAYOUT_GEIST = """import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";

export const metadata: Metadata = {
  title: "Create Next App",
  description: "Generated by create next app",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${GeistSans.variable} ${GeistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
"""


def _format_bedrock_error(exc: ClientError) -> str:
    err = exc.response.get("Error", {}) if exc.response else {}
    code = err.get("Code", "")
    msg = err.get("Message", str(exc))
    if code == "ThrottlingException":
        if "tokens per day" in msg.lower():
            return (
                "[bedrock] Límite diario de tokens agotado para esta cuenta.\n"
                "  → Esperá al reset, probá otra cuenta/región o contactá AWS Support.\n"
                f"  → Detalle: {msg}"
            )
        return f"[bedrock] Throttling ({code}): {msg}"
    if code == "ValidationException":
        return f"[bedrock] Validación: {msg}"
    if code == "AccessDeniedException":
        return f"[bedrock] Acceso denegado: {msg}"
    return f"[bedrock] {code}: {msg}"


def _llm_provider() -> str:
    p = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if p in ("ollama", "gemini", "bedrock", "groq"):
        return p
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return "gemini"
    return "bedrock"


def _build_llm_client():
    prov = _llm_provider()
    if prov == "ollama":
        ollama_llm.check_ollama_ready()
        return {
            "provider": "ollama",
            "model": ollama_llm.ollama_model_id(),
        }
    if prov == "groq":
        groq_llm.require_groq_key()
        return {
            "provider": "groq",
            "model": groq_llm.groq_model_id(),
        }
    if prov == "gemini":
        if not os.environ.get("GEMINI_API_KEY", "").strip():
            raise RuntimeError(
                "Modo Gemini: falta GEMINI_API_KEY. "
                "Alternativas gratis: LLM_PROVIDER=groq + GROQ_API_KEY, o LLM_PROVIDER=ollama"
            )
        return {
            "provider": "gemini",
            "model": ", ".join(gemini_llm.gemini_model_chain()),
        }
    return _build_bedrock()


def _llm_dispatch(client: Any, messages: list, system: str, tools: Optional[list]) -> Any:
    if isinstance(client, dict):
        prov = client.get("provider")
        if prov == "gemini":
            return gemini_llm.llm_gemini(client, messages, system, tools)
        if prov == "ollama":
            return ollama_llm.llm_ollama(client, messages, system, tools)
        if prov == "groq":
            return groq_llm.llm_groq(client, messages, system, tools)
    return llm(client, messages, system, tools=tools)


def _summarize_dispatch(client: Any, transcript: str, summarizer_system: str) -> str:
    if isinstance(client, dict):
        prov = client.get("provider")
        if prov == "gemini":
            return gemini_llm.gemini_text_only(client, transcript, summarizer_system)
        if prov == "ollama":
            return ollama_llm.ollama_text_only(client, transcript, summarizer_system)
        if prov == "groq":
            return groq_llm.groq_text_only(client, transcript, summarizer_system)
    return converse_text_only(client, transcript, summarizer_system)


def _build_bedrock():
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Definí AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY y AWS_DEFAULT_REGION en el entorno."
        )
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )


def _execute_code_headless_gui_blocked(code: str) -> Optional[str]:
    """El sandbox E2B no tiene X11; tkinter/GUIs fallan siempre."""
    low = code.lower()
    banned_substrings = (
        "tkinter",
        "pyqt5",
        "pyqt6",
        "pyside2",
        "pyside6",
        "wxpython",
        "import wx",
        "dearpygui",
    )
    if any(s in low for s in banned_substrings):
        return (
            "BLOQUEADO: el sandbox no tiene pantalla ($DISPLAY). tkinter/PyQt/wx no funcionan aquí. "
            "Implementá la UI en Next.js dentro de web-app/ (page.tsx, CSS/Tailwind) y usá subprocess "
            "para npm, no GUIs de escritorio."
        )
    return None


def _execute_code_blocking_dev_servers(code: str) -> Optional[str]:
    """npm start / next dev no terminan → timeout en E2B; el agente debe usar solo npm run build."""
    lower = code.lower()
    compact = re.sub(r"[\s\"']+", "", lower)
    needles = (
        "npm,start",
        "npm,run,start",
        "npm,run,dev",
        "next,dev",
        "yarn,dev",
        "pnpm,dev",
        "npx,next,dev",
        "npxnextdev",
    )
    if any(n in compact for n in needles):
        return (
            "BLOQUEADO: `npm start`, `npm run dev` y servidores similares no deben ejecutarse aquí: "
            "el proceso no termina y la herramienta hace timeout. "
            "Usá solo `subprocess.run(['npm', 'run', 'build'], cwd='web-app', timeout=600)` para verificar. "
            "En la respuesta final explicá que quien use el repo puede correr `npm run dev` localmente."
        )
    if "subprocess" in lower and (
        re.search(r"npm\s+run\s+dev\b", lower)
        or re.search(r"\bnpm\s+start\b", lower)
        or re.search(r"\bnext\s+dev\b", lower)
    ):
        return (
            "BLOQUEADO: comando de servidor de desarrollo detectado (no terminan). "
            "Usá `npm run build` con cwd en el proyecto Next.js."
        )
    return None


def _coerce_execute_code_if_raw_npm(code: str, work_dir: str) -> str:
    """
    Modelos suelen mandar ``npm run build`` suelto como ``code`` → SyntaxError.
    Lo convertimos a Python + subprocess con cwd en el proyecto Next.
    """
    s = code.strip()
    if not s:
        return code
    lines = [
        ln.strip()
        for ln in s.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        return code
    for ln in lines:
        low_ln = ln.lower()
        if ln.startswith("import ") or ln.startswith("from ") or ln.startswith("def "):
            return code
        if low_ln.startswith("pip ") or low_ln.startswith("python "):
            return code
    blob = " ".join(lines).lower()
    if "npm" not in blob or "build" not in blob:
        return code
    if not all(re.match(r"^npm(\s+|$)", ln.lower()) for ln in lines):
        return code
    cwd = work_dir.strip().strip("/") or "web-app"
    m = re.search(r"--prefix\s+([^\s#]+)", blob)
    if m:
        cwd = m.group(1).strip().strip("/").strip('"').strip("'")
    wd_lit = json.dumps(cwd)
    return (
        "import subprocess, sys, os\n"
        f"r = subprocess.run(\n"
        f'    ["npm", "run", "build"],\n'
        f"    cwd={wd_lit},\n"
        f'    env={{**os.environ, "CI": "1"}},\n'
        "    capture_output=True,\n"
        "    text=True,\n"
        "    timeout=900,\n"
        ")\n"
        'print("returncode=", r.returncode)\n'
        "sys.stdout.write((r.stdout or '')[-14000:])\n"
        "sys.stderr.write((r.stderr or '')[-14000:])\n"
    )


def _canonicalize_write_path_for_workdir(path: str, work_dir: str) -> str:
    """
    Evita escribir ``src/app/page.tsx`` en el home del sandbox en vez de dentro de ``web-app/``.
    """
    p = path.replace("\\", "/").strip()
    np = _normalize_rel_path(p)
    wd = work_dir.strip().strip("/") or "web-app"
    if np.startswith(wd + "/") or np.startswith("../"):
        return p
    if np in (wd, ".", ""):
        return p
    # Rutas típicas del esqueleto Next dentro del proyecto
    if (
        np.startswith("src/")
        or np.startswith("public/")
        or np == "package.json"
        or np.endswith("package-lock.json")
        or np.endswith(".tsx")
        or np.endswith(".ts")
        or np.endswith(".js")
        or np.endswith(".mjs")
        or np.endswith(".css")
        or np.endswith(".json")
    ):
        return f"{wd}/{np}"
    return p


def _compact_ws(low: str) -> str:
    """Une tokens ignorando cualquier espacio/salto (para detectar ``export default`` partido en líneas)."""
    return "".join(low.split())


def _execute_code_smells_like_react_or_fake_cli(code: str) -> Optional[str]:
    """
    Qwen a menudo manda TS/JS o pseudo-shell a ``execute_code`` → SyntaxError opaco en la línea ~10.
    Devolvé un mensaje claro o None si parece Python real.
    """
    s = code.strip()
    low = s.lower()
    compact = _compact_ws(low)
    if "write_file" in low and "-path" in low:
        return (
            "Pegaste una orden tipo **shell** (`write_file -path ...`) dentro de `execute_code`. "
            "Eso no es Python ni se ejecuta. Usá la herramienta **`write_file`** del asistente (JSON con "
            "`path` y `content`), no un bloque ```bash."
        )
    if "```" in s or "glob -pattern" in low or "read_file -path" in low.replace(" ", ""):
        if any(x in low for x in ("write_file", "glob ", "read_file", "list_directory")):
            return (
                "El código parece **comandos de mentira en markdown** mezclados con texto. `execute_code` "
                "solo acepta **Python** válido. Invocá **`glob`**, **`read_file`**, **`write_file`** como "
                "**tool calls** de la API, no dentro de ```bash."
            )
    if "usestate" in compact or "useeffect" in compact:
        return (
            "`execute_code` ejecuta **solo Python**. El código parece **React** (`useState`/hooks). "
            "Usá **`write_file`** o **`replace_in_file`** sobre `src/app/page.tsx` (y `'use client'` si hay estado)."
        )
    if "exportdefaultfunction" in compact or "importreact" in compact:
        return (
            "`execute_code` ejecuta **solo Python**. Detecté sintaxis **React/TSX**: usá **`write_file`** "
            "para `page.tsx`, no mezcles JSX en Python."
        )
    if "from'react'" in compact or 'from"react"' in compact or "from`react`" in compact:
        return (
            "`execute_code` ejecuta **solo Python**. Hay `import … from 'react'`: es **TS/JS**. "
            "Usá **`write_file`** en `web-app/src/app/page.tsx`."
        )
    if "{usestate" in compact or "usestate<string" in compact:
        return (
            "`execute_code` ejecuta **solo Python**. Detecté **useState** tipado (TS): usá **`write_file`**, "
            "no `execute_code`."
        )
    # JSX suelto
    if "</" in s and ">" in s and ("className=" in s or "classname=" in low):
        return (
            "`execute_code` ejecuta **solo Python**. Parece **JSX** (`className=` / tags). "
            "Usá **`write_file`** para `.tsx`."
        )
    return None


def _execute_code_sniff_after_syntax_error(code: str) -> Optional[str]:
    """Si ``compile()`` falló, último intento de explicar que era TSX mezclado con texto."""
    low = code.lower()
    c = _compact_ws(low)
    if "usestate" in c or "exportdefault" in c or "from'react'" in c or 'from"react"' in c:
        return (
            "El error de sintaxis suele ser **TypeScript/React pegado en `execute_code`**. "
            "Para **build** usá solo la línea `npm run build` o Python con `subprocess.run(...)`; "
            "el componente va en **`write_file`**."
        )
    if "ejecuta" in low and "comando" in low and "npm" in low:
        return (
            "No pegues instrucciones en español + comando en el mismo `code`. "
            "Para build: **solo** `npm run build` (una línea) o el snippet Python de `subprocess`."
        )
    return None


def _sandbox_run_code_or_raise(sbx: Sandbox, code: str) -> Any:
    """Envuelve ``run_code`` para mensajes claros si E2B corta por timeout/red."""
    try:
        return sbx.run_code(code)
    except BaseException as e:
        if sbx_tools.is_e2b_transport_error(e):
            raise RuntimeError(sbx_tools.E2B_TRANSPORT_HINT) from e
        raise


def execute_code(sbx: Sandbox, code: str) -> dict[str, Any]:
    block = _execute_code_headless_gui_blocked(code)
    if block:
        return {"results": [], "logs": [], "errors": [block]}
    block_srv = _execute_code_blocking_dev_servers(code)
    if block_srv:
        return {"results": [], "logs": [], "errors": [block_srv]}
    full = sbx_tools.with_home_as_cwd(code)
    full = _coerce_execute_code_if_raw_npm(full, _default_work_dir())
    confused = _execute_code_smells_like_react_or_fake_cli(full)
    if confused:
        return {"results": [], "logs": [], "errors": [f"BLOQUEADO: {confused}"]}
    try:
        compile(full, "<execute_code>", "exec")
    except SyntaxError as e:
        hint = _execute_code_sniff_after_syntax_error(full)
        base = (
            f"SyntaxError (código no ejecutado en sandbox): {e}. "
            "Revisá que sea **Python** (import subprocess, etc.); TSX/React va en **write_file**, no aquí."
        )
        if hint:
            base = base + " " + hint
        return {"results": [], "logs": [], "errors": [base]}
    try:
        execution = sbx.run_code(full)
    except BaseException as e:
        if sbx_tools.is_e2b_transport_error(e):
            return {
                "results": [],
                "logs": [],
                "errors": [f"{sbx_tools.E2B_TRANSPORT_HINT} ({type(e).__name__})"],
            }
        raise
    results: list[str] = []
    for r in execution.results:
        if hasattr(r, "text") and r.text:
            results.append(r.text)
    errors: list[str] = []
    if execution.error:
        errors.append(f"{execution.error.name}: {execution.error.value}")
    logs: list[str] = []
    if execution.logs.stdout:
        logs.extend(execution.logs.stdout)
    if execution.logs.stderr:
        errors.extend(execution.logs.stderr)

    errs = list(errors)
    joined_r = " ".join(str(x) for x in results)
    if "returncode=1" in joined_r or "returncode: 1" in joined_r:
        low = full.lower()
        if "npm" in low and "build" in low:
            tail = "\n".join((logs or [])[-100:])
            if tail.strip():
                errs.append(
                    "--- Fragmento de salida de `npm run build` / next (buscá Error / Failed / Syntax) ---\n"
                    + tail[-16000:]
                )
            errs.append(
                "El build terminó con código de error. Si editaste `web-app/page.tsx` con HTML, esa ruta "
                "no es la del App Router: usá **`web-app/src/app/page.tsx`** con componente React (JSX), no `<!DOCTYPE`."
            )
    return {"results": results, "logs": logs, "errors": errs}


def _normalize_rel_path(p: str) -> str:
    return p.replace("\\", "/").strip().lstrip("./")


def _write_file_nextjs_guard(path: str, content: Any) -> Optional[dict[str, str]]:
    """
    Corta el bucle infinito: modelo escribe HTML en ``web-app/page.tsx`` (Next lo ignora;
    la app real está en ``src/app/page.tsx``) y ``npm run build`` falla sin que el LLM entienda por qué.
    """
    if not isinstance(content, str):
        return None
    np = _normalize_rel_path(path)
    wd = _default_work_dir().strip().strip("/") or "web-app"
    if np == f"{wd}/page.tsx":
        return {
            "error": (
                f"Ruta incorrecta: con `--src-dir` la página del App Router es **`{wd}/src/app/page.tsx`**, "
                f"no `{wd}/page.tsx` (ese archivo no alimenta la ruta `/`). "
                "Usá `read_file` sobre `src/app/page.tsx` y editá ahí con React/TSX."
            )
        }
    if np.endswith(".tsx"):
        head = content.lstrip()[:1500].lower()
        if "<!doctype" in head or (head.startswith("<html") and "export default" not in content[:3000]):
            return {
                "error": (
                    "No escribas un documento HTML completo (`<!DOCTYPE`, `<html>`) en un `.tsx` de Next. "
                    "Usá solo JSX dentro de `export default function Page() { return ( ... ); }` "
                    "(y `'use client'` si usás useState)."
                )
            }
    return None


def execute_tool(name: str, args: str, tools: dict[str, Callable], **kwargs: Any) -> dict[str, Any]:
    try:
        args_dict = json.loads(args)
        if name == "write_file":
            pth = str(args_dict.get("path", "")).replace("\\", "/").strip()
            pth = _canonicalize_write_path_for_workdir(pth, _default_work_dir())
            args_dict["path"] = pth
            content_raw = args_dict.get("content")
            wf_guard = _write_file_nextjs_guard(pth, content_raw)
            if wf_guard is not None:
                return wf_guard
            if isinstance(content_raw, str):
                pl = pth.lower()
                if pl.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
                    head = content_raw.lstrip().lower()[:800]
                    if head.startswith("<svg") or "<svg" in head[:400]:
                        return {
                            "error": (
                                "El contenido parece **SVG/XML** pero la extensión es de imagen raster "
                                f"(`{pth}`). Guardalo como `.svg` o usá un PNG binario real; "
                                "un `.png` con texto SVG rompe `<Image>` / el navegador."
                            )
                        }
            if pth.endswith("package.json") and "node_modules/" not in pth:
                raw = args_dict.get("content", "")
                if isinstance(raw, str):
                    try:
                        json.loads(raw)
                    except json.JSONDecodeError as e:
                        return {
                            "error": (
                                "package.json debe ser JSON válido (npm no acepta YAML ni claves sin comillas). "
                                f"JSONDecodeError: {e}"
                            )
                        }
        if name not in tools:
            return {"error": f"Herramienta '{name}' no existe."}
        result = tools[name](**args_dict, **kwargs)
    except json.JSONDecodeError as e:
        return {"error": f"{name}: argumentos JSON inválidos: {e}"}
    except sbx_tools.ToolError as e:
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Argumento faltante: {e}"}
    except Exception as e:
        return {"error": str(e)}
    return result if isinstance(result, dict) else {"result": result}


MAX_SYSTEM_NUDGES = 5


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _e2b_kill_running_sandboxes_before_start() -> None:
    """
    Opcional: elimina **todos** los sandboxes en ejecución asociados a ``E2B_API_KEY``.
    Útil para liberar el límite de sandboxes concurrentes o dejar el dashboard vacío.

    Activar: ``E2B_KILL_RUNNING_SANDBOXES=1`` (¡mata también otros procesos que usen la misma key!).
    """
    if not _env_truthy("E2B_KILL_RUNNING_SANDBOXES"):
        return
    try:
        infos = Sandbox.list()
    except Exception as e:
        print(f"[e2b] E2B_KILL_RUNNING_SANDBOXES: no se pudo listar sandboxes: {e}", flush=True)
        return
    n = 0
    for info in infos:
        sid = getattr(info, "sandbox_id", None) or str(info)
        try:
            if Sandbox.kill(sid):
                n += 1
        except Exception as e:
            print(f"[e2b] kill {sid!r} falló: {e}", flush=True)
    print(
        f"[e2b] E2B_KILL_RUNNING_SANDBOXES: terminados {n}/{len(infos)} sandbox(es) que estaban LIVE.",
        flush=True,
    )


def _bootstrap_workspace(sbx: Sandbox, work_dir: str) -> None:
    """Crea la carpeta de trabajo en el sandbox sin pasar por el LLM (evita rutas inexistentes)."""
    if os.environ.get("SKIP_SANDBOX_BOOTSTRAP", "").lower() in ("1", "true", "yes"):
        return
    wd = json.dumps(work_dir)
    reset = _env_truthy("SANDBOX_RESET_WORKDIR")
    reset_py = "True" if reset else "False"
    code = f"""
import json, os, shutil
root = {wd}
if {reset_py}:
    if os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
os.makedirs(root, exist_ok=True)
readme = os.path.join(root, "README_SANDBOX.txt")
if not os.path.isfile(readme):
    with open(readme, "w", encoding="utf-8") as f:
        f.write("Workspace del agente. El proyecto Next.js debe generarse en esta carpeta.\\n")
print(json.dumps({{"bootstrap": True, "work_dir": root}}))
"""
    execution = _sandbox_run_code_or_raise(sbx, sbx_tools.with_home_as_cwd(code))
    err_parts: list[str] = []
    if execution.error:
        err_parts.append(f"{execution.error.name}: {execution.error.value}")
    if execution.logs.stderr:
        err_parts.extend(execution.logs.stderr)
    if err_parts:
        print(f"[e2b] bootstrap: {'; '.join(err_parts)}", flush=True)
    else:
        extra = " (SANDBOX_RESET_WORKDIR: carpeta recreada)" if reset else ""
        print(f"[e2b] Workspace listo: ./{work_dir}/{extra}", flush=True)


def _bootstrap_nextjs_in_workdir(sbx: Sandbox, work_dir: str) -> None:
    """
    Si no hay ``package.json`` con Next válido, deja ``work_dir`` **vacío** (incl. README),
    ejecuta ``create-next-app@14.2.18`` con ``CI=1`` y cwd en esa carpeta.

    Si dejábamos ``README_SANDBOX.txt``, la carpeta no estaba vacía y el CLI entraba en el
    wizard ("What is your project named?") sin generar un ``package.json`` válido.
    """
    if os.environ.get("SKIP_NEXT_BOOTSTRAP", "").lower() in ("1", "true", "yes"):
        print("[e2b] Next bootstrap omitido (SKIP_NEXT_BOOTSTRAP).", flush=True)
        return
    root_literal = json.dumps(work_dir)
    stdin_feed = os.environ.get("NEXT_BOOTSTRAP_STDIN", "\n" * 60)
    stdin_literal = json.dumps(stdin_feed)
    code = (
        """
import json, os, shutil, subprocess
root = """
        + root_literal
        + """
abs_root = os.path.abspath(root)


def _valid_package_json():
    pj = os.path.join(abs_root, "package.json")
    if not os.path.isfile(pj):
        return False
    try:
        with open(pj, "r", encoding="utf-8") as f:
            d = json.load(f)
        deps = d.get("dependencies")
        return isinstance(deps, dict) and "next" in deps
    except Exception:
        return False


def _hoist_nested_next_project():
    # CNA a veces crea un subdirectorio (p. ej. my-app); el agente espera package.json en work_dir.
    if _valid_package_json():
        return True
    if not os.path.isdir(abs_root):
        return False
    candidates = []
    for name in os.listdir(abs_root):
        if name.startswith("."):
            continue
        sub = os.path.join(abs_root, name)
        if not os.path.isdir(sub):
            continue
        pj = os.path.join(sub, "package.json")
        if not os.path.isfile(pj):
            continue
        try:
            with open(pj, "r", encoding="utf-8") as f:
                d = json.load(f)
            deps = d.get("dependencies")
            if isinstance(deps, dict) and "next" in deps:
                candidates.append(sub)
        except Exception:
            continue
    if len(candidates) != 1:
        return False
    nested = candidates[0]
    for item in os.listdir(nested):
        shutil.move(os.path.join(nested, item), os.path.join(abs_root, item))
    shutil.rmtree(nested, ignore_errors=True)
    return _valid_package_json()


if _valid_package_json():
    print(json.dumps({"next_bootstrap": "skipped", "reason": "package.json Next válido"}))
else:
    try:
        parent = os.path.dirname(abs_root) or "."
        os.makedirs(parent, exist_ok=True)
        shutil.rmtree(abs_root, ignore_errors=True)
        os.makedirs(abs_root, exist_ok=True)
    except Exception as e:
        print(json.dumps({"next_bootstrap": "error", "reason": "clean_failed", "detail": str(e)}))
    else:
        env = dict(os.environ)
        env["CI"] = "1"
        env["npm_config_yes"] = "true"
        env["TERM"] = "dumb"
        env["FORCE_COLOR"] = "0"
        env["NO_COLOR"] = "1"
        cmd = [
            "npx", "create-next-app@14.2.18", "--yes", ".",
            "--typescript", "--tailwind", "--eslint", "--app", "--src-dir",
            "--no-import-alias", "--use-npm",
        ]
        stdin_feed = """
        + stdin_literal
        + """
        # Con ``input=`` no uses ``stdin=PIPE`` (Python 3: ambos juntos → ValueError).
        r = subprocess.run(
            cmd,
            cwd=abs_root,
            env=env,
            input=stdin_feed,
            capture_output=True,
            text=True,
            timeout=900,
        )
        tail_out = (r.stdout or "")[-2500:]
        tail_err = (r.stderr or "")[-2500:]
        if r.returncode != 0:
            print(json.dumps({
                "next_bootstrap": "error",
                "returncode": r.returncode,
                "stdout_tail": tail_out,
                "stderr_tail": tail_err,
            }))
        else:
            if not _valid_package_json():
                _hoist_nested_next_project()
            if not _valid_package_json():
                print(json.dumps({
                    "next_bootstrap": "error",
                    "returncode": r.returncode,
                    "reason": "invalid_package_json_after_create",
                    "stdout_tail": tail_out,
                    "stderr_tail": tail_err,
                }))
            else:
                try:
                    rm = os.path.join(abs_root, "README_SANDBOX.txt")
                    with open(rm, "w", encoding="utf-8") as f:
                        f.write("Proyecto Next.js generado por bootstrap del agente.\\n")
                except Exception:
                    pass
                print(json.dumps({"next_bootstrap": "ok", "returncode": r.returncode}))
"""
    )
    execution = _sandbox_run_code_or_raise(sbx, sbx_tools.with_home_as_cwd(code))
    if execution.error:
        print(
            f"[e2b] next bootstrap: error sandbox: {execution.error.name}: {execution.error.value}",
            flush=True,
        )
        return
    out = "".join(execution.logs.stdout or [])
    info = None
    for raw in reversed(out.strip().splitlines()):
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            info = json.loads(raw)
            break
        except json.JSONDecodeError:
            continue
    if info is None:
        print(f"[e2b] next bootstrap: sin JSON en salida: {out[-400:]!r}", flush=True)
        return
    status = info.get("next_bootstrap")
    if status == "ok":
        print("[e2b] Next.js 14 bootstrap OK en ./{}/".format(work_dir), flush=True)
    elif status == "skipped":
        print(f"[e2b] Next bootstrap: omitido ({info.get('reason', '')}).", flush=True)
    else:
        print(f"[e2b] next bootstrap: {info}", flush=True)


def _patch_nextjs_geist_stack(sbx: Sandbox, work_dir: str) -> None:
    """
    Tras ``create-next-app``, sustituye fuentes locales (``.woff`` + ``next/font/local``) por el paquete
    publicado ``geist``, añade ``transpilePackages: ['geist']`` en ``next.config.*`` y borra
    ``src/app/fonts``. Evita el fallo de compilación ``Cannot read properties of undefined (reading 'ascent')``.

    Desactivar: ``SKIP_GEIST_PATCH=1``.
    """
    if _env_truthy("SKIP_GEIST_PATCH"):
        print("[e2b] Parche Geist omitido (SKIP_GEIST_PATCH).", flush=True)
        return
    wd = work_dir.strip().strip("/") or "web-app"
    root_literal = json.dumps(wd)
    layout_literal = repr(_NEXT_LAYOUT_GEIST)
    code = (
        """
import json, os, shutil, subprocess
root = """
        + root_literal
        + """
abs_root = os.path.abspath(root)
layout_tsx = """
        + layout_literal
        + """


def _patch_next_config(path):
    try:
        with open(path, encoding="utf-8") as f:
            t = f.read()
    except OSError:
        return False
    if "transpilePackages" in t and "geist" in t:
        return True
    if "const nextConfig = {}" not in t:
        return False
    new_block = 'const nextConfig = {\\n  transpilePackages: ["geist"],\\n}'
    t = t.replace("const nextConfig = {}", new_block, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(t)
    return True


result = {"geist_patch": "skipped", "reason": "no_package_json"}
pj = os.path.join(abs_root, "package.json")
if not os.path.isfile(pj):
    print(json.dumps(result))
else:
    try:
        with open(pj, encoding="utf-8") as f:
            pkg = json.load(f)
        deps = pkg.get("dependencies") if isinstance(pkg, dict) else None
        if not isinstance(deps, dict) or "next" not in deps:
            result = {"geist_patch": "skipped", "reason": "not_a_next_app"}
        else:
            env = dict(os.environ)
            env["CI"] = "1"
            env["npm_config_yes"] = "true"
            r = subprocess.run(
                ["npm", "install", "geist", "--save"],
                cwd=abs_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if r.returncode != 0:
                result = {
                    "geist_patch": "npm_install_failed",
                    "stderr_tail": (r.stderr or "")[-1800:],
                }
            else:
                lp = os.path.join(abs_root, "src", "app", "layout.tsx")
                os.makedirs(os.path.dirname(lp), exist_ok=True)
                with open(lp, "w", encoding="utf-8") as f:
                    f.write(layout_tsx)
                cfg_ok = False
                for name in ("next.config.mjs", "next.config.js", "next.config.ts"):
                    cfg = os.path.join(abs_root, name)
                    if os.path.isfile(cfg):
                        cfg_ok = _patch_next_config(cfg) or cfg_ok
                fonts_dir = os.path.join(abs_root, "src", "app", "fonts")
                had_fonts = os.path.isdir(fonts_dir)
                if had_fonts:
                    shutil.rmtree(fonts_dir, ignore_errors=True)
                result = {
                    "geist_patch": "ok",
                    "next_config_updated": cfg_ok,
                    "removed_local_fonts_dir": had_fonts,
                }
    except Exception as e:
        result = {"geist_patch": "error", "detail": str(e)}
    print(json.dumps(result))
"""
    )
    try:
        execution = _sandbox_run_code_or_raise(sbx, sbx_tools.with_home_as_cwd(code))
    except BaseException as e:
        print(f"[e2b] Parche Geist: error al ejecutar en sandbox: {e}", flush=True)
        return
    if execution.error:
        print(
            f"[e2b] Parche Geist: sandbox: {execution.error.name}: {execution.error.value}",
            flush=True,
        )
        return
    out = "".join(execution.logs.stdout or [])
    info = None
    for raw in reversed(out.strip().splitlines()):
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            info = json.loads(raw)
            break
        except json.JSONDecodeError:
            continue
    if info is None:
        print(f"[e2b] Parche Geist: sin JSON en salida: {out[-400:]!r}", flush=True)
        return
    if info.get("geist_patch") == "ok":
        extra = []
        if info.get("removed_local_fonts_dir"):
            extra.append("fuentes locales eliminadas")
        if info.get("next_config_updated"):
            extra.append("next.config actualizado")
        print(
            "[e2b] Parche Geist aplicado (paquete `geist`, layout.tsx, sin src/app/fonts)"
            + (f" — {', '.join(extra)}" if extra else ""),
            flush=True,
        )
    elif info.get("geist_patch") == "npm_install_failed":
        print(
            f"[e2b] Parche Geist: npm install geist falló — {info.get('stderr_tail', '')!r}",
            flush=True,
        )
    elif info.get("reason") == "not_a_next_app":
        print("[e2b] Parche Geist: omitido (package.json sin dependencia `next`).", flush=True)
    elif info.get("geist_patch") == "skipped":
        print(f"[e2b] Parche Geist: omitido ({info.get('reason', '')}).", flush=True)
    elif info.get("geist_patch") == "error":
        print(f"[e2b] Parche Geist: error — {info.get('detail', info)}", flush=True)


def _sandbox_has_valid_next_in_workdir(sbx: Sandbox, work_dir: str) -> bool:
    """
    True solo si existe ``{work_dir}/package.json`` legible y con dependencia ``next``.
    Evita dar por bueno un package.json suelto en la raíz del sandbox sin app en web-app.
    """
    wd = work_dir.strip().strip("/") or "web-app"
    path = f"{wd}/package.json"
    try:
        txt = sbx_tools.read_file(sbx, path, limit=64_000)
        data = json.loads(txt)
    except (sbx_tools.ToolError, json.JSONDecodeError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    deps = data.get("dependencies")
    if not isinstance(deps, dict):
        return False
    return "next" in deps


def _print_sandbox_project_location(sandbox_id: str, work_dir: str) -> None:
    """Aclara que page.tsx no está en el repo local; evita confusiones con Finder/VSCode."""
    wd = work_dir.strip().strip("/") or "web-app"
    print(
        "[e2b] Ubicación del código generado:\n"
        f"  · Solo en la VM E2B, **mismo sandbox que este id** (copialo y buscalo en e2b.dev):\n"
        f"      {sandbox_id}\n"
        f"  · La carpeta `{wd}/` **no existe** hasta que este script imprima «Workspace listo: ./{wd}/». "
        "Si abriste un sandbox desde el dashboard **sin** tener corriendo `agent_web_dev.py`, "
        "/home/user solo tendrá .bashrc, etc. — es normal.\n"
        f"  · Cuando exista, en la plantilla code-interpreter la ruta típica es **/code/{wd}/** "
        f"(el runtime hace chdir a `/code` si existe). Si tu imagen no tiene `/code`, quedará en "
        f"**/home/user/{wd}/**. En Filesystem → Go: `/code/{wd}` o `/home/user/{wd}`.\n"
        "  · No confundas con la carpeta `/code` vacía **antes** de que corra el agente.\n"
        "  · Copia local al terminar: SANDBOX_EXPORT_DIR=./e2b-export\n"
        "  · Si el script ya cerró el sandbox, esos archivos ya no están en E2B.",
        flush=True,
    )


def _text_asks_user_instead_of_tools(text: str) -> bool:
    if not (text and text.strip()):
        return False
    t = text.lower()
    needles = (
        "¿en qué archivo",
        "ruta del archivo",
        "proporciones",
        "proporcioname",
        "compartí el",
        "comparte el",
        "comparte la",
        "necesito que me",
        "no tengo acceso",
        "no puedo ver",
        "sin acceso a",
        "no tengo la estructura",
        "¿puedes compartir",
        "¿podés compartir",
        "¿puedes proporcionar",
    )
    return any(n in t for n in needles)


def _text_looks_like_unknown_tool_call(text: str, valid_tools: set[str]) -> bool:
    """
    Modelos locales a veces escriben JSON con ``name`` que no es una herramienta real
    (p. ej. ``create-next-app``); no se coerciona → el agente creía que la tarea terminó.
    """
    if not (text and text.strip()):
        return False
    raw = text.strip()
    low = raw.lower()
    if '"name"' not in raw and "'name'" not in raw:
        return False
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if isinstance(name, str) and name and name not in valid_tools:
            return True
    # Atajos frecuentes si el JSON está truncado o mal formado
    for fake in (
        "create-next-app",
        "createnextapp",
        "create_next_app",
        "next-create",
        "run_terminal",
    ):
        if f'"{fake}"' in low or f"'{fake}'" in low:
            return True
    return False


KNOWN_TOOL_NAMES = frozenset(
    {
        "execute_code",
        "list_directory",
        "read_file",
        "write_file",
        "search_file_content",
        "replace_in_file",
        "glob",
    }
)


def _last_user_instruction_text(messages: list) -> str:
    """Último mensaje `user` con texto libre (ignora bloques que solo traen resultados de tools)."""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        parts: list[str] = []
        if isinstance(c, list):
            for block in c:
                if isinstance(block, dict) and "text" in block:
                    tx = block.get("text")
                    if isinstance(tx, str) and tx.strip():
                        parts.append(tx)
        elif isinstance(c, str) and c.strip():
            parts.append(c)
        out = "\n".join(parts).strip()
        if out:
            return out
    return ""


def _text_premature_prose_no_tools(
    text: str, work_dir: str, last_user_instruction: str
) -> bool:
    """
    El modelo “cierra” con charla o tutoriales sin invocar herramientas (Ollama suele hacerlo).
    Sin esto, el bucle imprime «tarea completada» aunque no haya tocado el repo ni el build.
    """
    if not (text and text.strip()):
        return False
    t = text.lower()
    u = last_user_instruction.lower()
    wd = work_dir.lower()

    if "no es directamente posible" in t:
        return True
    if "en su lugar" in t and ("html" in t or "index.html" in t):
        return True
    if ("aplicación web simple" in t or "aplicacion web simple" in t) and ("html" in t or "css" in t):
        return True
    if "aquí tienes un ejemplo" in t or "aqui tienes un ejemplo" in t:
        return True
    if "estructura del proyecto" in t and "index.html" in t:
        return True
    if "```json" in t and '"name"' in t:
        for n in KNOWN_TOOL_NAMES:
            if f'"name": "{n}"' in text or f'"name":"{n}"' in text:
                return True
    # Indica pasos / plan largo sin ejecutar (típico de qwen en CPU)
    if re.search(r"^###\s*paso\s*\d", t, re.MULTILINE) and t.count("###") >= 2:
        return True
    # Pasa trabajo al usuario en lugar de editar el sandbox
    if ("asegúrate" in t or "asegurate" in t) and (
        "index.html" in t
        or "tu archivo" in t
        or "vos " in t
        or "tú debes" in t
        or "tu debes" in t
    ):
        return True
    if "referenciado en tu archivo" in t or "referenciado en el archivo" in t:
        return True
    # Íconos / nav en Next: no debe mandar a index.html suelto
    if "index.html" in t and "page.tsx" not in t and "src/app" not in t:
        if wd in t or "nav" in u or "ícono" in u or "icono" in u or "navbar" in u:
            return True
    # Pega React en fences como sustituto de write_file (no crea archivos en el sandbox)
    if "```jsx" in t or "```tsx" in t or "```javascript" in t or "```js" in t:
        if (
            "usestate" in t
            or "export default function" in t
            or "'use client'" in t
            or '"use client"' in text
        ):
            return True
    # Simula herramientas como comandos bash (glob/read_file/write_file -path …)
    if "```bash" in t or "```shell" in t:
        if any(
            fake in t
            for fake in (
                "write_file",
                "read_file",
                "glob ",
                "glob -",
                "list_directory",
                "execute_code",
            )
        ):
            return True
    # Disculpa por error de herramienta pero sigue sin llamarlas; a menudo viene con código en markdown
    if "lo siento" in t and ("sintaxis" in t or "syntaxerror" in t) and "```" in text:
        return True
    # Qwen usa ```typescript además de tsx/jsx
    if "```typescript" in t or ("```ts" in t and "```tsx" not in t):
        if "usestate" in t or "export default function" in t:
            return True
    # CSS en markdown en vez de escribir `globals.css` / layout con tools
    if "```css" in t:
        if any(
            k in t
            for k in (
                "globals",
                "navbar",
                ".nav",
                "body {",
                "background",
                "box-shadow",
                "src/app",
            )
        ):
            return True
    # Cierra la tarea pero le pide al usuario que corra el build (el agente debe usar execute_code)
    if "npm run build" in t or "`npm run build`" in text.lower():
        if any(
            x in t
            for x in (
                "puedes ejecutar",
                "podés ejecutar",
                "ahora puedes",
                "deberías ejecutar",
                "puedes correr",
                "podés correr",
                "te toca ejecutar",
                "ejecuta el siguiente comando",
                "ejecutá el siguiente comando",
            )
        ):
            return True
    return False


TOOLS: dict[str, Callable[..., Any]] = {
    "execute_code": execute_code,
    "list_directory": sbx_tools.list_directory,
    "read_file": sbx_tools.read_file,
    "write_file": sbx_tools.write_file,
    "search_file_content": sbx_tools.search_file_content,
    "replace_in_file": sbx_tools.replace_in_file,
    "glob": sbx_tools.glob,
}

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "name": "execute_code",
        "description": (
            "Ejecuta código Python en el sandbox (subprocess para npm/npx). "
            "El código se valida con compile() antes de enviarlo: si hay SyntaxError verás el error en el resultado. "
            "Para create-next-app: versión fija create-next-app@14.2.18, flag --yes tras el nombre del paquete, "
            "y subprocess.run(..., env=dict(os.environ, CI='1'), stdin=subprocess.DEVNULL) para no colgar en prompts."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Código Python válido"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_directory",
        "description": "Lista archivos y carpetas en una ruta del sandbox.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Lee un archivo de texto del sandbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "description": "Máx. caracteres (opcional)"},
                "offset": {"type": "integer", "description": "Desde byte/carácter (opcional)"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": (
            "Escribe o sobrescribe un archivo en el sandbox. "
            "Para `package.json` el contenido debe ser JSON válido (rechazado si no parsea)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_file_content",
        "description": "Busca un patrón regex en archivos de texto; devuelve matches paginados.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "max_results": {"type": "integer", "default": 50},
                "root": {"type": "string", "description": "Directorio raíz de búsqueda", "default": "."},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "replace_in_file",
        "description": "Reemplaza la primera ocurrencia de old por new en un archivo.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
            "required": ["path", "old", "new"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "glob",
        "description": "Lista archivos por patrón glob desde una raíz (ej. **/*.tsx).",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "root": {"type": "string", "default": "."},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
]


def run_agent(
    query: str,
    messages: Optional[list] = None,
    *,
    client: Optional[Any] = None,
    sbx: Optional[Sandbox] = None,
    system: Optional[str] = None,
    max_steps: int = 30,
) -> tuple[list, str]:
    """
    Ejecuta el agente; muta `messages` si se pasa la misma lista entre llamadas.
    Retorna (messages, última_respuesta_texto).

    Si ``system`` es None, se usa la plantilla con `SANDBOX_PROJECT_DIR` (default `web-app`).
    """
    if messages is None:
        messages = []
    work_dir = _default_work_dir()
    if system is None:
        system = system_prompt_for_workdir(work_dir)

    if client is None:
        client = _build_llm_client()
        if isinstance(client, dict):
            prov = client.get("provider")
            if prov == "gemini":
                print(f"[gemini] modelo: {client.get('model')}", flush=True)
            elif prov == "ollama":
                _print_ollama_startup(
                    str(client.get("model") or ""),
                    ollama_llm.ollama_base_url(),
                )
            elif prov == "groq":
                _print_groq_startup(str(client.get("model") or ""))
            else:
                print(f"[bedrock] modelo en uso: {bedrock_model_id()}", flush=True)
        else:
            print(f"[bedrock] modelo en uso: {bedrock_model_id()}", flush=True)

    if isinstance(client, dict) and client.get("provider") == "ollama":
        system = system + ollama_llm.ollama_tool_use_system_reminder()

    if sbx is None:
        key = os.environ.get("E2B_API_KEY")
        if not key:
            raise RuntimeError("Definí E2B_API_KEY en el entorno.")
        os.environ["E2B_API_KEY"] = key
        print("[e2b] Creando sandbox (suele tardar unos segundos la primera vez)…", flush=True)
        # e2b-code-interpreter ≥1.x: sandbox síncrono con Sandbox(), no Sandbox.create()
        sbx = Sandbox(timeout=60 * 60)
        print("[e2b] Sandbox listo.", flush=True)

    print(f"[e2b] sandbox_id={sbx.sandbox_id}", flush=True)
    _print_sandbox_project_location(sbx.sandbox_id, work_dir)
    _bootstrap_workspace(sbx, work_dir)
    _bootstrap_nextjs_in_workdir(sbx, work_dir)
    _patch_nextjs_geist_stack(sbx, work_dir)

    summarizer_system = (
        "Resumís conversaciones de desarrollo con precisión técnica. "
        "Solo el resumen, en español, sin saludos."
    )

    def summarizer(transcript: str) -> str:
        return _summarize_dispatch(client, transcript, summarizer_system)

    user_text = query
    if messages:
        user_text = (
            "[Seguimiento: mismo sandbox y archivos que en el historial. "
            "No pidas rutas al usuario. Para UI/nav/íconos: glob/read_file/search_file_content "
            "y luego editá.]\n\n"
            + query
        )
    messages.append({"role": "user", "content": [{"text": user_text}]})
    steps = 0
    last_output = ""
    nudge_count = 0
    # (path normalizado, sha256 del contenido) — evita bucles del LLM reenviando el mismo write_file
    write_sig_history: Deque[Tuple[str, str]] = deque(maxlen=24)

    while steps < max_steps:
        compress_kw: dict = {}
        if isinstance(client, dict) and client.get("provider") == "groq":
            compress_kw["max_tokens_before_compress"] = _groq_compress_token_limit()
        elif isinstance(client, dict) and client.get("provider") == "ollama":
            compress_kw["max_tokens_before_compress"] = _ollama_compress_token_limit()
        compressed = maybe_compress(messages, summarizer, **compress_kw)
        if compressed is not messages:
            messages[:] = compressed
        if steps == 0:
            if isinstance(client, dict):
                cp = client.get("provider")
                if cp == "gemini":
                    print("[gemini] Primer turno (generateContent)…", flush=True)
                elif cp == "ollama":
                    print("[ollama] Primer turno (/api/chat, puede tardar en CPU)…", flush=True)
                elif cp == "groq":
                    print("[groq] Primer turno (chat/completions)…", flush=True)
                else:
                    print(
                        "[bedrock] Enviando primer turno a Converse (puede tardar; Bedrock reintenta si hay throttling)…",
                        flush=True,
                    )
            else:
                print(
                    "[bedrock] Enviando primer turno a Converse (puede tardar; Bedrock reintenta si hay throttling)…",
                    flush=True,
                )
        response = _llm_dispatch(client, messages, system, TOOL_SCHEMAS)
        print(f"\n[paso {steps}]")
        has_tool_call = False
        tool_results: list[dict] = []
        headless_display_error_nudge = False

        messages.append(response.bedrock_message())

        for part in response.output:
            if part.type == "message":
                last_output = part.content
                print(f"[agente] {part.content[:500]}{'…' if len(part.content) > 500 else ''}")
            elif part.type == "function_call":
                has_tool_call = True
                name = part.name
                print(f"[agente][{name}] …")
                if name == "write_file":
                    try:
                        ad_w = (
                            json.loads(part.arguments)
                            if isinstance(part.arguments, str)
                            else part.arguments
                        )
                    except (json.JSONDecodeError, TypeError):
                        ad_w = {}
                    pth_w = str(ad_w.get("path", "")).replace("\\", "/").strip()
                    content_w = ad_w.get("content")
                    if isinstance(content_w, str) and pth_w:
                        npw = _normalize_rel_path(pth_w)
                        digest = hashlib.sha256(
                            content_w.encode("utf-8", errors="replace")
                        ).hexdigest()
                        if (npw, digest) in write_sig_history:
                            result = {
                                "error": (
                                    "BLOQUEADO — `write_file` **repetido** (mismo path y mismo contenido que un paso "
                                    "anterior que ya se aplicó bien). No reenvíes el mismo archivo: corré "
                                    f"`npm run build` con `cwd='{work_dir}/'` y arreglá solo lo que el log indique; "
                                    "si importás desde `src/app/page.tsx` hacia `src/components/`, usá "
                                    "`../components/Nombre`, no `./components/Nombre`."
                                )
                            }
                            print(
                                "[agente] aviso: write_file idéntico a uno ya grabado — usar build o cambiar el diff",
                                flush=True,
                            )
                        else:
                            result = execute_tool(
                                name, part.arguments, TOOLS, sbx=sbx
                            )
                            if isinstance(result, dict) and not result.get("error"):
                                write_sig_history.append((npw, digest))
                    else:
                        result = execute_tool(
                            name, part.arguments, TOOLS, sbx=sbx
                        )
                else:
                    result = execute_tool(name, part.arguments, TOOLS, sbx=sbx)
                if isinstance(client, dict) and client.get("provider") == "groq":
                    result = _truncate_tool_result(
                        result, _groq_tool_result_max_chars(), "Groq"
                    )
                elif isinstance(client, dict) and client.get("provider") == "ollama":
                    result = _truncate_tool_result(
                        result, _ollama_tool_result_max_chars(), "Ollama"
                    )
                if name == "execute_code" and isinstance(result, dict):
                    for err in result.get("errors") or []:
                        el = str(err).lower()
                        if "tclerror" in el or "no display" in el or "$display" in el:
                            headless_display_error_nudge = True
                            break
                print(f"[{name}] → {str(result)[:300]}…" if len(str(result)) > 300 else f"[{name}] → {result}")
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": part.call_id,
                            "name": name,
                            "content": [{"json": result}],
                        }
                    }
                )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
            if headless_display_error_nudge:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": (
                                    "[Instrucción del sistema — no la ignores] "
                                    "El sandbox E2B no tiene entorno gráfico ($DISPLAY). "
                                    "No reintentes tkinter. Construí la lista de tareas estilo Windows 95 "
                                    f"como **página web** en `{work_dir}/` (componentes React + estilos CSS/Tailwind) "
                                    "y verificá con `npm run build` usando subprocess con `cwd` en esa carpeta."
                                )
                            }
                        ],
                    }
                )
                print("[agente] refuerzo: sin pantalla en sandbox — usar Next.js, no tkinter", flush=True)

        if not has_tool_call:
            missing_project = not _sandbox_has_valid_next_in_workdir(sbx, work_dir)
            deflects = _text_asks_user_instead_of_tools(last_output)
            unknown_tool_json = _text_looks_like_unknown_tool_call(last_output, set(TOOLS.keys()))
            last_instr = _last_user_instruction_text(messages)
            premature_prose = _text_premature_prose_no_tools(
                last_output, work_dir, last_instr
            )
            if (
                missing_project
                or deflects
                or unknown_tool_json
                or premature_prose
            ) and nudge_count < MAX_SYSTEM_NUDGES:
                nudge_count += 1
                chunks: list[str] = []
                if unknown_tool_json:
                    chunks.append(
                        "[Instrucción del sistema — no la ignores] "
                        "Tu salida parece una llamada a una **herramienta que no existe**. "
                        "Las únicas herramientas son: **execute_code**, **list_directory**, **read_file**, "
                        "**write_file**, **search_file_content**, **replace_in_file**, **glob**. "
                        "No existen tools `create-next-app`, `create_next_app` ni similares: usá **execute_code** "
                        f"con `npx` o **write_file** bajo `{work_dir}/` (Next ya suele estar creado por el runtime)."
                    )
                if missing_project:
                    chunks.append(
                        "[Instrucción del sistema — no la ignores] "
                        f"No hay Next.js válido en `{work_dir}/package.json` (falta archivo o dependencia `next`). "
                        f"No des por terminada la tarea. En el siguiente paso usá herramientas: "
                        f'`list_directory(".")`, `list_directory("{work_dir}")`, luego `execute_code` '
                        "con Python válido para ejecutar `create-next-app@14.2.18` **dentro** de "
                        f'`{work_dir}` (`cwd="{work_dir}"`). Pasá `env=dict(os.environ, CI="1")` y '
                        '`stdin=subprocess.DEVNULL` para evitar el wizard interactivo. Argumentos: '
                        '`["npx", "create-next-app@14.2.18", "--yes", ".", "--typescript", "--tailwind", '
                        '"--eslint", "--app", "--src-dir", "--no-import-alias", "--use-npm"]`. Sin shell=True.'
                    )
                if deflects:
                    chunks.append(
                        "[Instrucción del sistema — no la ignores] "
                        "No pidas rutas ni fragmentos al usuario: el proyecto está en el sandbox. "
                        f'Usá `glob` y `read_file` sobre `.` y `{work_dir}` en el siguiente paso.'
                    )
                if premature_prose:
                    chunks.append(
                        "[Instrucción del sistema — no la ignores] "
                        "No des por cerrada la tarea con tutoriales, listas de pasos solo en texto, "
                        "ni mandando al usuario a editar `index.html` suelto: el stack es **Next.js** en "
                        f"`{work_dir}/src/app/` (p. ej. `page.tsx`, `layout.tsx`). "
                        "En el **siguiente mensaje** invocá herramientas reales: `read_file` / `replace_in_file` / "
                        "`write_file` y validá con `execute_code` + `npm run build` (cwd en el proyecto). "
                        "**Prohibido** inventar ` ```bash ` con `write_file -path` o `glob -pattern`: eso **no ejecuta**. "
                        "Solo valen **function calls** del API (Ollama tool calling). "
                        "`execute_code` = **solo Python** (`subprocess`, etc.); el TSX va en **`write_file`**."
                    )
                messages.append({"role": "user", "content": [{"text": "\n\n".join(chunks)}]})
                print(
                    f"\n[agente] refuerzo automático ({nudge_count}/{MAX_SYSTEM_NUDGES}) — "
                    "se requiere seguir con herramientas",
                    flush=True,
                )
                continue

            if nudge_count >= MAX_SYSTEM_NUDGES:
                print(
                    "\n[agente] fin sin éxito claro: el modelo dejó de invocar herramientas "
                    f"tras {MAX_SYSTEM_NUDGES} refuerzos — revisá si hubo errores arriba (p. ej. SyntaxError, solo texto/```jsx).",
                    flush=True,
                )
            else:
                print("\n[agente] tarea completada (sin más herramientas)", flush=True)
            break

        steps += 1
    else:
        print(f"\n[agente] límite de {max_steps} pasos")

    return messages, last_output


def _execution_last_stdout_json(execution: Any) -> Any:
    """Última línea JSON en stdout de ``run_code`` (una sola línea emitida por el snippet)."""
    text = "".join(execution.logs.stdout or []).strip()
    if not text:
        raise ValueError("stdout vacío")
    last = text.splitlines()[-1]
    return json.loads(last)


def _sandbox_workdir_paths_for_export(sbx: Sandbox, work_dir: str) -> list[str]:
    """
    Lista rutas relativas (p. ej. web-app/src/...) bajo el cwd del sandbox, excluyendo
    node_modules, .next, .git, etc. Archivos > 12 MiB se omiten en el listado.
    """
    wd = work_dir.strip().strip("/") or "web-app"
    wd_lit = json.dumps(wd)
    max_b = 12_000_000
    code = f"""
import json, os
wd = {wd_lit}
skip_dirs = {{"node_modules", ".next", ".git", "__pycache__", ".venv", "venv", "dist", "build", "coverage"}}
out = []
if os.path.isdir(wd):
    for root, dirs, files in os.walk(wd):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for name in files:
            p = os.path.join(root, name)
            try:
                if not os.path.isfile(p):
                    continue
                if os.path.getsize(p) > {max_b}:
                    continue
                out.append(p.replace("\\\\", "/"))
            except OSError:
                continue
print(json.dumps(sorted(set(out))))
"""
    execution = _sandbox_run_code_or_raise(sbx, sbx_tools.with_home_as_cwd(code))
    if execution.error:
        raise RuntimeError(
            f"{execution.error.name}: {execution.error.value}"
        )
    data = _execution_last_stdout_json(execution)
    if not isinstance(data, list):
        raise TypeError(f"listado export: tipo {type(data)}")
    return [str(p) for p in data]


def _export_sandbox_workdir(sbx: Sandbox, work_dir: str, dest: Path) -> None:
    """
    Copia el árbol del proyecto Next desde el sandbox a disco local (sin node_modules ni .next).
    Activar con ``SANDBOX_EXPORT_DIR`` (ruta absoluta o relativa al cwd).

    Tras exportar podés borrar el sandbox en E2B y trabajar solo en tu máquina:
    ``npm install`` y ``npm run dev`` dentro de la carpeta del proyecto.
    """
    wd = work_dir.strip().strip("/") or "web-app"
    root = dest.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    fallback = [
        f"{wd}/package.json",
        f"{wd}/package-lock.json",
        f"{wd}/tsconfig.json",
        f"{wd}/src/app/page.tsx",
        f"{wd}/src/app/layout.tsx",
        f"{wd}/src/app/globals.css",
        f"{wd}/next.config.js",
        f"{wd}/next.config.mjs",
    ]
    try:
        paths = _sandbox_workdir_paths_for_export(sbx, wd)
    except (RuntimeError, ValueError, TypeError, json.JSONDecodeError) as e:
        print(f"[e2b] Export: listado recursivo falló ({e}); uso lista mínima.", flush=True)
        paths = []
    if not paths:
        paths = list(fallback)
    n = 0
    skipped = 0
    for rel in paths:
        try:
            body = sbx_tools.read_file(sbx, rel, limit=None)
        except sbx_tools.ToolError:
            skipped += 1
            continue
        out_path = root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body, encoding="utf-8")
        n += 1
    guide = root / "LOCAL_RUN.txt"
    proj_rel = wd
    proj_abs = root / proj_rel
    guide.write_text(
        "Proyecto exportado desde el sandbox E2B (sin node_modules ni .next).\n\n"
        f"1. cd {proj_abs}\n"
        "2. npm install\n"
        "3. npm run dev   (desarrollo)\n\n"
        "Producción local: npm run build && npm run start\n"
        "(sin build previo, next start falla: falta carpeta .next)\n\n"
        "Abrí http://localhost:3000\n\n"
        "Podés cerrar o borrar el sandbox en e2b.dev: el código ya está acá.\n"
        f"Atajo: ./scripts/run-exported-next.sh {proj_abs}\n",
        encoding="utf-8",
    )
    print(
        f"[e2b] Exportados {n} archivos → {root} "
        f"({skipped} omitidos por error de lectura). "
        f"Leé {guide.name} o corré scripts/run-exported-next.sh.",
        flush=True,
    )


def main() -> None:
    # Tarea Final — Parte 4: mismo historial entre tareas (conversaciones largas + compresión).
    historial: list = []
    try:
        client = _build_llm_client()
    except RuntimeError as e:
        print(str(e), flush=True)
        raise SystemExit(1) from e
    if isinstance(client, dict):
        mp = client.get("provider")
        if mp == "gemini":
            print(f"[gemini] modelo: {client.get('model')}", flush=True)
        elif mp == "ollama":
            _print_ollama_startup(
                str(client.get("model") or ""),
                ollama_llm.ollama_base_url(),
            )
        elif mp == "groq":
            _print_groq_startup(str(client.get("model") or ""))
        else:
            print(f"[bedrock] modelo en uso: {bedrock_model_id()}", flush=True)
    else:
        print(f"[bedrock] modelo en uso: {bedrock_model_id()}", flush=True)
    key = os.environ.get("E2B_API_KEY")
    if not key:
        raise RuntimeError("Definí E2B_API_KEY en el entorno.")
    os.environ["E2B_API_KEY"] = key
    _e2b_kill_running_sandboxes_before_start()
    print("[e2b] Creando sandbox (suele tardar unos segundos la primera vez)…", flush=True)
    # ``with`` llama a ``kill()`` al salir (normal o error); si no, los sandboxes quedan LIVE hasta el timeout.
    with Sandbox(timeout=60 * 60) as sbx:
        print("[e2b] Sandbox listo.", flush=True)
        exp_dir = os.environ.get("SANDBOX_EXPORT_DIR", "").strip()
        if exp_dir:
            print(
                f"[e2b] Al terminar se exportará el árbol de `{_default_work_dir()}/` (sin node_modules/.next) "
                f"a {exp_dir!r}. Luego: ./scripts/run-exported-next.sh {exp_dir.rstrip('/')}/{_default_work_dir()}",
                flush=True,
            )
        try:
            run_agent(
                "Crea una app de lista de tareas (agregar y quitar ítems) en Next.js. "
                "Prioridad 1: **mínimo que compile** — idealmente todo en `src/app/page.tsx`, "
                "corré `npm run build` antes de complicar. Prioridad 2 (después del build OK): "
                "estilo visual tipo Windows 95.",
                messages=historial,
                client=client,
                sbx=sbx,
            )
            wd = _default_work_dir()
            run_agent(
                f"Mejorá la visibilidad del nav (íconos o enlaces que se confunden con el fondo) en `{wd}/` "
                "usando **solo código**: `src/app/page.tsx`, `layout.tsx`, `globals.css` o estilos inline. "
                "Prioridad: **TypeScript/CSS que compile**; corré `npm run build`. "
                "No te quedes buscando `.svg` en rutas inventadas: si no hay assets, resolvelo con CSS/contraste.",
                messages=historial,
                client=client,
                sbx=sbx,
            )
        except ClientError as e:
            print(_format_bedrock_error(e), flush=True)
            raise SystemExit(1) from e
        except RuntimeError as e:
            if isinstance(client, dict) and client.get("provider") in ("gemini", "ollama", "groq"):
                print(str(e), flush=True)
                raise SystemExit(1) from e
            raise
        finally:
            if exp_dir:
                try:
                    _export_sandbox_workdir(sbx, _default_work_dir(), Path(exp_dir))
                except Exception as exc:
                    print(f"[e2b] Exportación SANDBOX_EXPORT_DIR falló: {exc}", flush=True)


if __name__ == "__main__":
    main()
