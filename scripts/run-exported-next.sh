#!/usr/bin/env bash
# Ejecuta en tu máquina el Next exportado con SANDBOX_EXPORT_DIR.
# Uso (desde la raíz del repo):
#   ./scripts/run-exported-next.sh ./e2b-export/web-app
set -euo pipefail
TARGET="${1:?Pasá la ruta a la carpeta del proyecto (ej. ./e2b-export/web-app)}"
if [[ ! -d "$TARGET" ]]; then
  echo "No existe el directorio: $TARGET" >&2
  exit 1
fi
cd "$TARGET"
if [[ ! -f package.json ]]; then
  echo "No hay package.json en $TARGET — ¿exportaste con SANDBOX_EXPORT_DIR?" >&2
  exit 1
fi
npm install
exec npm run dev
