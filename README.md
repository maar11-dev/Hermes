# 📚 Hermes — Agente RAG de Apuntes

> Chatea con tus apuntes en PDF usando inteligencia artificial local o cloud.

> Versión actual: motor RAG híbrido con streaming, reranking y frontend renovado.

<p align="center">
  <img src="frontend/hermes.png" alt="Hermes" width="180">
</p>

![Hermes](https://img.shields.io/badge/RAG-ChromaDB-orange)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20Anthropic-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-yellow)

---

## ✨ Características

| Funcionalidad         | Descripción                                              |
|-----------------------|----------------------------------------------------------|
| 📄 **Subida de PDFs** | Arrastra y suelta archivos o selecciónalos desde la barra lateral; se indexan automáticamente |
| 💬 **Chat en streaming** | Las respuestas aparecen poco a poco para una experiencia más fluida |
| 🔎 **Búsqueda híbrida** | Combina BM25 + vectorial con Reciprocal Rank Fusion |
| 🎯 **Reranking**      | Reordena los fragmentos recuperados con un cross-encoder |
| 🧠 **Historial**      | Mantiene contexto de conversación en varias preguntas |
| 📋 **Resumen**        | Genera resúmenes estructurados al instante                |
| ✍️ **Apuntes**        | Convierte el contenido en apuntes tipo Cornell/Zettelkasten |
| 🎯 **Test**           | Genera preguntas tipo test para repasar                   |
| 🎛️ **Frontend renovado** | Barra lateral con documentos, contador, toasts y selector múltiple |
| 🔒 **100% local**     | Los documentos nunca salen de tu máquina (con Ollama)     |
| 💾 **Persistencia**   | La base vectorial se guarda en disco entre sesiones       |

---

## 🚀 Instalación rápida

### Requisitos previos
- **Python 3.10+** — [python.org](https://python.org)
- **Ollama** (recomendado) — [ollama.com](https://ollama.com)  
  *Alternativa: API key de Anthropic (ver configuración)*

### 1 — Clona el repositorio

```bash
git clone https://github.com/tu-usuario/Hermes.git
cd Hermes
```

### 2 — Arranca (Linux / macOS)

```bash
chmod +x start.sh
./start.sh
```

### 2 — Arranca (Windows)

```
start.bat
```

> El script crea automáticamente el entorno virtual, instala las dependencias, descarga el modelo de Ollama si falta y arranca el servidor.

### 3 — Abre el navegador

```
http://localhost:8000
```

---

## ⚙️ Configuración

Edita el archivo `.env` (se crea automáticamente en el primer arranque):

```dotenv
# LLM a usar: "ollama" (local) o "anthropic" (cloud)
LLM_PROVIDER=ollama

# ── Ollama ──────────────────────────────
OLLAMA_MODEL=gemma2:2b               # o tinyllama, qwen2.5, phi3...
OLLAMA_BASE_URL=http://localhost:11434

# ── Anthropic (alternativa) ─────────────
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# ── Embeddings ──────────────────────────
EMBED_MODEL=all-MiniLM-L6-v2   # ~90MB, se descarga una vez

# ── Parámetros RAG ──────────────────────
CHUNK_SIZE=300     # palabras por fragmento
CHUNK_OVERLAP=60   # solapamiento
TOP_K=4            # fragmentos recuperados
```

### Modelos Ollama recomendados

| Modelo         | Comando                   | VRAM   | Notas                    |
|----------------|---------------------------|--------|--------------------------|
| `gemma2:2b`    | `ollama pull gemma2:2b`   | ~0.5 GB| Muy ligero en CPU/VRAM   |
| `tinyllama`    | `ollama pull tinyllama`   | <1 GB  | Mínimo CPU, recomendado |
| `phi3`         | `ollama pull phi3`        | ~2 GB  | Balance CPU/calidad      |
| `qwen2.5`      | `ollama pull qwen2.5`     | ~2-3 GB| Muy bueno en español     |
| `llama3.2`     | `ollama pull llama3.2`    | ~4 GB  | Mejor calidad, más CPU   |

### Novedades recientes

- Motor RAG con búsqueda híbrida BM25 + vectorial y fusión RRF.
- Reordenación de resultados con `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Chat por streaming desde `/api/chat`.
- Historial de conversación en la API para mantener contexto.
- Frontend reorganizado con estilos propios, logo y mejor experiencia de uso.

---

## 📁 Estructura del proyecto

```
Hermes/
├── backend/
│   ├── main.py           # Servidor FastAPI + endpoints API
│   ├── rag_engine.py     # Lógica RAG (orquestador + streaming a LLMs)
│   ├── requirements.txt
│   └── RAG/              # Componentes RAG (embedder, indexer, retriever, vector store)
│       ├── __init__.py
│       ├── embedder.py
│       ├── indexer.py
│       ├── retriever.py
│       └── vector_store.py
├── frontend/
│   ├── index.html        # Interfaz web
│   ├── styles.css        # Estilos de la interfaz
│   └── hermes.png        # Logo usado en el README y la UI
├── data/                 # Base de datos vectorial (autogenerado)
│   ├── chroma_db/
│   └── documents.json
├── .env                  # Tu configuración local (no subas al repo)
├── .env.example          # Plantilla de configuración
├── start.sh              # Arranque Linux/macOS
├── start.bat             # Arranque Windows
└── README.md
```

---

## 🔌 API REST

El backend expone los siguientes endpoints:

| Método   | Ruta                        | Descripción                     |
|----------|-----------------------------|---------------------------------|
| `GET`    | `/api/health`               | Estado del servidor             |
| `GET`    | `/api/documents`            | Listar documentos indexados     |
| `POST`   | `/api/documents`            | Subir y indexar un PDF          |
| `DELETE` | `/api/documents/{id}`       | Eliminar un documento           |
| `POST`   | `/api/chat`                 | Consulta RAG en streaming       |

Ejemplo de consulta:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Qué es la fotosíntesis?", "doc_ids": [], "history": []}'
```

---

## 🧩 Cómo funciona

```
PDF → Extracción de texto (PyMuPDF)
    → Fragmentación en chunks con solapamiento
  → Recuperación híbrida BM25 + embeddings
  → Fusión de resultados + reranking
  → Almacenamiento en ChromaDB (disco)

Consulta → Embedding de la pregunta
     → Búsqueda vectorial + BM25
     → Reciprocal Rank Fusion + cross-encoder
     → Recuperación de los TOP_K fragmentos más relevantes
         → Prompt con contexto → LLM (Ollama / Anthropic)
     → Respuesta al usuario en streaming
```

---

## 🛠️ Solución de problemas

**El servidor no arranca**
- Asegúrate de tener Python 3.10+: `python3 --version`

**Ollama no responde**
- Comprueba que está en marcha: `ollama serve`
- Verifica que el modelo está descargado: `ollama list`

**Respuestas lentas**
- Normal en la primera consulta (carga el modelo de embeddings)
- Considera usar un modelo Ollama más pequeño (`phi3`)
- O usa Anthropic API para respuestas más rápidas

**El PDF no se indexa**
- Asegúrate de que el PDF tiene texto (no es solo imágenes escaneadas)
- Para PDFs escaneados, considera usar OCR previo (ej. `ocrmypdf`)

---

## 📄 Licencia

MIT — libre para usar, modificar y distribuir.
