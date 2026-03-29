# Agente web (tarea)

Agente que genera una app **Next.js** en un sandbox **E2B** usando un LLM (Bedrock, Gemini, Groq u Ollama).

## Tecnologías

- **Python 3.10+** (recomendado), `pip`, entorno virtual
- **boto3**, **e2b-code-interpreter**, **python-dotenv**
- **AWS Bedrock** (opcional) · **Google Gemini** · **Groq** · **Ollama** (el proveedor se elige con variables de entorno)
- En el sandbox: **Next.js 14**, **React**, **TypeScript**, **Tailwind CSS**, fuentes **Geist**

## Cómo correrlo

1. Cloná o abrí el repo y creá el entorno:

   ```bash
   cd fullAgentDev
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   python -m pip install -U pip setuptools wheel
   pip install -r requirements.txt
   ```

2. Creá un archivo **`.env`** en la raíz del repo con al menos:

   - **`E2B_API_KEY`** — obligatoria ([e2b.dev](https://e2b.dev))
   - Según el LLM que uses, por ejemplo:
     - Bedrock: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`
     - Gemini: `GEMINI_API_KEY` (y opcional `LLM_PROVIDER=gemini`)
     - Groq: `GROQ_API_KEY`, `LLM_PROVIDER=groq`
     - Ollama: `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=...` (con `ollama serve` en marcha)

3. Ejecutá el agente:

   ```bash
   python tarea/agent_web_dev.py
   ```

4. **Exportar** el proyecto Next a tu máquina (opcional):

   ```bash
   export SANDBOX_EXPORT_DIR=./e2b-export
   python tarea/agent_web_dev.py
   ```

   Luego, en la carpeta exportada (p. ej. `e2b-export/web-app`): `npm install` y `npm run dev`. Podés usar `./scripts/run-exported-next.sh ./e2b-export/web-app`.

## Producción local del Next exportado

```bash
cd ruta/al/web-app
npm install
npm run build
npm run start
```

Sin `build` previo, `npm run start` falla (falta `.next`).
