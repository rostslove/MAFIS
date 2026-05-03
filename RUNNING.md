# Running MultiAgentFedot.IndustrialSystem (MAFIS) with Fedot.Industrial 0.5.0

`fedot-ind==0.5.0` is not installed from PyPI in this project. Use a local checkout of `Fedot.Industrial` and its Poetry environment.

Expected folder layout:

```text

  Fedot.Industrial/
  industrial-learning-agent/
```

## 1. Prepare Fedot.Industrial

```powershell
git clone https://github.com/aimclub/Fedot.Industrial.git
cd Fedot.Industrial
poetry install
```

If the repository is already cloned, just run:

```powershell
cd ~/Fedot.Industrial
poetry install
```

## 2. Restore The Poetry Env If Requirements Were Installed Before


```bash
cd ~/Fedot.Industrial
poetry install --sync
```

If your Poetry version says `--sync` is deprecated, use:

```bash
poetry sync
```

## 3. Install Backend App Dependencies Into That Poetry Env

```powershell
cd D:\Diploma\Fedot.Industrial
poetry run pip install -r ../industrial-learning-agent/backend/requirements.txt
```

On Linux use `/`, not Windows `\`:

```bash
cd ~/Fedot.Industrial
poetry run pip install -r ../industrial-learning-agent/backend/requirements.txt
```

The backend requirements intentionally do not install `fedot-ind`, `pandas`, `scikit-learn`, `xgboost`, FastAPI, or the official `mcp` package. Fedot.Industrial owns the scientific stack and Starlette, while the backend uses a local MCP tool adapter. This avoids conflicts with `python-fasthtml -> starlette~=1.0` and `spacy -> pydantic<2`.

## 4. Run Local LLM

Recommended default model:

```text
qwen2.5-coder:14b
```

It is the shared model for all MAFIS agents and gives more stable JSON/tool reasoning than the previous 7B default. If the server is CPU-only or low on RAM, use `qwen2.5-coder:7b` or `qwen2.5-coder:3b` instead.

With Docker Compose, Ollama is started automatically and the model is pulled by `ollama-pull`:

```bash
cd ~/industrial-learning-agent
docker compose up --build
```

If the server has NVIDIA GPU runtime configured:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The Ollama API is bound only to server-local `127.0.0.1:11434`. Backend containers call it through `http://ollama:11434/v1`.

## 5. Run Backend Without Docker

```powershell
cd D:\Diploma\Fedot.Industrial
$env:FEDOT_INDUSTRIAL_PATH="$HOME/Fedot.Industrial"
$env:LLM_BASE_URL="http://localhost:11434/v1"
$env:LLM_API_KEY="ollama"
$env:LLM_MODEL="qwen2.5-coder:14b"
poetry run python ../industrial-learning-agent/backend/app.py
```

Backend URL:

```text
http://localhost:8001
```

You can also put `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` into the project `.env`; the backend reads it from the project root even when launched from the Fedot.Industrial directory.

Linux:

```bash
cd ~/Fedot.Industrial
export FEDOT_INDUSTRIAL_PATH="$HOME/Fedot.Industrial"
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_API_KEY="ollama"
export LLM_MODEL="qwen2.5-coder:14b"
poetry run python ../industrial-learning-agent/backend/app.py
```

## 6. Run Frontend

Open a second terminal:

```powershell
cd ~/industrial-learning-agent/frontend
python -m venv .venv
./.venv/Scripts/bin/activate
pip install -r requirements.txt
$env:BACKEND_URL="http://localhost:8001"
streamlit run streamlit_app.py --server.port 8502
```

Frontend URL:

```text
http://localhost:8502
```

## Docker Notes

Docker build expects `Fedot.Industrial` next to this project. The compose file uses:

```yaml
additional_contexts:
  fedot_industrial: ${FEDOT_INDUSTRIAL_CONTEXT:-../Fedot.Industrial}
```

If the checkout is elsewhere:

```powershell
$env:FEDOT_INDUSTRIAL_CONTEXT="$HOME/Fedot.Industrial"
docker compose up --build
```

The backend image installs Fedot.Industrial from that local checkout with Poetry.
