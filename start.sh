#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  Hermes — Script de arranque (Linux / macOS)
# ──────────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
DATA_DIR="$SCRIPT_DIR/data"

echo ""
echo "  ╔═══════════════════════════════════════╗"
echo "  ║   📚  Hermes — Agente RAG         ║"
echo "  ╚═══════════════════════════════════════╝"
echo ""

# ── Crear .env si no existe ───────────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/.env" ]; then
  cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
  echo "  ✓ .env creado. Puedes editarlo para cambiar el LLM."
fi

# ── Copiar .env al directorio backend ─────────────────────────────────────
cp "$SCRIPT_DIR/.env" "$BACKEND_DIR/.env" 2>/dev/null || true

# ── Crear entorno virtual si no existe ───────────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "  → Creando entorno virtual Python..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ── Instalar dependencias ─────────────────────────────────────────────────
echo "  → Instalando dependencias (puede tardar la primera vez)..."
pip install -q --upgrade pip
pip install -q -r "$BACKEND_DIR/requirements.txt"

# ── Directorio de datos ───────────────────────────────────────────────────
mkdir -p "$DATA_DIR"
rm -f "$DATA_DIR/.browser_opened"

# ── Utilidad: leer variable desde .env de forma robusta ──────────────────
read_env() {
  awk -F= -v key="$1" '$1==key {v=$2} END {gsub(/[[:space:]]/, "", v); print v}' "$SCRIPT_DIR/.env"
}

# ── Verificar Ollama (si se usa) ──────────────────────────────────────────
LLM_PROVIDER="$(read_env LLM_PROVIDER)"
if [ -z "$LLM_PROVIDER" ]; then
  LLM_PROVIDER="ollama"
fi

if [ "$LLM_PROVIDER" = "ollama" ] || [ -z "$LLM_PROVIDER" ]; then
  if ! command -v ollama &>/dev/null; then
    echo ""
    echo "  ⚠️  Ollama no encontrado."
    echo "     Instálalo en https://ollama.com o cambia LLM_PROVIDER=anthropic en .env"
    echo ""
  else
    OLLAMA_MODEL="$(read_env OLLAMA_MODEL)"
    if [ -z "$OLLAMA_MODEL" ]; then
      OLLAMA_MODEL="gemma2:2b"
    fi

    echo "  → Comprobando modelo Ollama: $OLLAMA_MODEL"
    if ! ollama list | grep -q "$OLLAMA_MODEL"; then
      echo "  → Descargando modelo $OLLAMA_MODEL (solo la primera vez)..."
      ollama pull "$OLLAMA_MODEL"
    fi
    if ! pgrep -x "ollama" > /dev/null; then
      echo "  → Arrancando servidor Ollama en segundo plano..."
      ollama serve &>/dev/null &
      sleep 2
    fi
  fi
fi

# ── Arrancar FastAPI ──────────────────────────────────────────────────────
echo ""
echo "  ✓ Servidor iniciado → http://localhost:8000"
echo "  El navegador se abrira cuando el servidor este listo"
echo "  (Ctrl+C para detener)"
echo ""

cd "$BACKEND_DIR"
"$VENV_DIR/bin/uvicorn" main:app --host 0.0.0.0 --port 8000 --reload
