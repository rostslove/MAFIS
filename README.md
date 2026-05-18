# MultiAgentFedot.IndustrialSystem (MAFIS)

MAFIS is a graph-first multi-agent system for building, training, validating, and explaining
Fedot.Industrial pipelines. LLM agents propose pipeline graphs, execute them through a local
MCP-style tool adapter, analyze runtime diagnostics, and report the best evaluated result.

The project is focused on tabular and time-series machine learning workflows backed by
Fedot.Industrial 0.5.0.

## What MAFIS Does

- Profiles CSV datasets and exposes a compact data summary to the Architect agent.
- Lets the Architect propose a valid Fedot/Fedot.Industrial graph.
- Lets the user inspect, edit, and approve the graph before evaluation.
- Trains the approved graph through Fedot.Industrial.
- Lets Engineer and Critic surface metrics, failures, recovery attempts, and graph feedback.
- Generates a final Scribe report in the UI.
- Supports explicit `train.csv` / `test.csv` paths for reproducible benchmark folds.

Supported task types:

- `classification`
- `regression`
- `ts_classification`
- `ts_regression`
- `ts_forecasting`

## Runtime Layout

```text
Streamlit UI (:8502)
        |
        | HTTP / SSE
        v
Starlette backend (:8001)
        |
        | local MCP-style tool adapter
        v
Graph/profile/train/report tools
        |
        v
Fedot.Industrial + FEDOT + sklearn
```

Ollama provides the OpenAI-compatible LLM endpoint used by the agents.

## Requirements

- Docker and Docker Compose
- A local checkout of `Fedot.Industrial` 0.5.0
- Ollama model, by default `qwen2.5-coder:32b`
- Optional: NVIDIA container runtime for GPU-backed Ollama

Expected sibling layout:

```text
Fedot.Industrial/
industrial-learning-agent/
```

The backend intentionally uses Fedot.Industrial from a local source checkout instead of installing
`fedot-ind` from PyPI. This keeps the scientific dependency stack aligned with
`Fedot.Industrial/poetry.lock`.

## Quick Start With Docker

From the repository root:

```bash
docker compose up --build
```

Open:

```text
Frontend: http://localhost:8502
Backend:  http://localhost:8001
Ollama:   http://localhost:11434
```

For NVIDIA GPU runtime:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

If `Fedot.Industrial` is not located at `../Fedot.Industrial`, set:

```bash
export FEDOT_INDUSTRIAL_CONTEXT=/path/to/Fedot.Industrial
docker compose up --build
```

On PowerShell:

```powershell
$env:FEDOT_INDUSTRIAL_CONTEXT="D:\Diploma\Fedot.Industrial"
docker compose up --build
```

## Local Backend Run

Prepare Fedot.Industrial first:

```bash
cd ../Fedot.Industrial
poetry install
poetry run pip install -r ../industrial-learning-agent/backend/requirements.txt
```

Run the backend from the Fedot.Industrial Poetry environment:

```bash
cd ../Fedot.Industrial
export FEDOT_INDUSTRIAL_PATH="$PWD"
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_API_KEY="ollama"
export LLM_MODEL="qwen2.5-coder:32b"
poetry run python ../industrial-learning-agent/backend/app.py
```

PowerShell:

```powershell
cd D:\Diploma\Fedot.Industrial
$env:FEDOT_INDUSTRIAL_PATH="D:\Diploma\Fedot.Industrial"
$env:LLM_BASE_URL="http://localhost:11434/v1"
$env:LLM_API_KEY="ollama"
$env:LLM_MODEL="qwen2.5-coder:32b"
poetry run python ..\industrial-learning-agent\backend\app.py
```

## Local Frontend Run

In a second terminal:

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export BACKEND_URL="http://localhost:8001"
streamlit run streamlit_app.py --server.port 8502
```

PowerShell:

```powershell
cd frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:BACKEND_URL="http://localhost:8001"
streamlit run streamlit_app.py --server.port 8502
```

## Configuration

The backend reads `.env` from the project root when present.

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_MODEL` | `qwen2.5-coder:32b` | Ollama model used by all agents. |
| `LLM_BASE_URL` | `http://localhost:11434/v1` locally, `http://ollama:11434/v1` in Docker | OpenAI-compatible LLM endpoint. |
| `LLM_API_KEY` | `ollama` | Placeholder key for Ollama-compatible API calls. |
| `OLLAMA_CONTEXT_LENGTH` | `32768` | Ollama context window. |
| `FEDOT_INDUSTRIAL_PATH` | `../Fedot.Industrial` or Docker image path | Local Fedot.Industrial source checkout. |
| `FEDOT_INDUSTRIAL_CONTEXT` | `../Fedot.Industrial` | Docker BuildKit additional context path. |
| `FEDOT_N_JOBS` | `0` | Training worker count. `0` means use available CPUs. |
| `TRAINING_PROGRESS_LOG_INTERVAL` | `30` | Seconds between long-running training progress log messages. |

## Using Datasets

The UI supports three dataset entry points:

1. Upload a CSV file.
2. Load a CSV by path from the shared `data/` directory.
3. Load a registered benchmark dataset.

For reproducible benchmarks, pass train and test files separately. The backend keeps the explicit
split instead of creating a new hold-out split:

```json
{
  "csv_path": "data/datasets/example/fold_0/train.csv",
  "test_csv_path": "data/datasets/example/fold_0/test.csv",
  "target_column": "target",
  "task_type": "regression"
}
```

The UI preview intentionally renders only a small sample of rows and columns. Full dataset
statistics are computed on the training split and passed to the Architect.

## API Endpoints

- `GET /health`
- `GET /config`
- `GET /tools`
- `POST /datasets/path/load`
- `POST /benchmarks/load`
- `POST /architect/chat`
- `POST /architect/revise`
- `POST /graph/mutate`
- `POST /orchestrate`
- `POST /orchestrate/stream`

The frontend uses `/orchestrate/stream` to receive live agent, training, and report events.

## Project Structure

```text
backend/
  agents/          LLM agent implementations and schemas
  benchmarks/      benchmark registry and dataset loaders
  mcp/
    mcp_client.py  local MCP-style client used by agents
    mcp_server.py  profiling, training, validation, explanation, report tools
  utils/           shared backend utilities
  app.py           Starlette HTTP/SSE application
  graph_engine.py  graph schema, validation, mutation, conversion, metrics
frontend/
  streamlit_app.py interactive graph-first UI
data/
  shared local/Docker data volume
```

Additional docs:

- [Architecture](ARCHITECTURE.md)
- [Running Guide](RUNNING.md)

## Development Checks

Basic syntax check:

```bash
python -m py_compile backend/app.py backend/orchestrator.py backend/graph_engine.py backend/mcp/mcp_server.py
```

When working with the local Fedot.Industrial Poetry environment:

```bash
cd ../Fedot.Industrial
poetry run python -m py_compile ../industrial-learning-agent/backend/app.py
```

## Notes

- The official Python MCP package is not installed in the Fedot.Industrial environment because of
  dependency conflicts. The project uses a local MCP-style adapter.
- FastAPI is not used for the same reason; the backend uses Starlette directly.
- The Docker backend mounts `./data` into `/app/data`, so files loaded by the frontend and backend
  must be visible in that shared volume.
