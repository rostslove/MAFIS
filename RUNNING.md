# Running GraphAutoML with Fedot.Industrial 0.5.0

`fedot-ind==0.5.0` is not installed from PyPI in this project. Use a local checkout of `Fedot.Industrial` and its Poetry environment.

Expected folder layout:

```text
Diploma/
  Fedot.Industrial/
  industrial-learning-agent/
```

## 1. Prepare Fedot.Industrial

```powershell
cd D:\Diploma
git clone https://github.com/aimclub/Fedot.Industrial.git
cd Fedot.Industrial
poetry install
```

If the repository is already cloned, just run:

```powershell
cd D:\Diploma\Fedot.Industrial
poetry install
```

## 2. Restore The Poetry Env If Requirements Were Installed Before

If you previously ran the old backend requirements, restore Fedot.Industrial dependencies first. The old file pinned `pandas` and `scikit-learn` too low for Fedot.Industrial 0.5.0.

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
poetry run pip install -r ..\industrial-learning-agent\backend\requirements.txt
```

On Linux use `/`, not Windows `\`:

```bash
cd ~/Fedot.Industrial
poetry run pip install -r ../industrial-learning-agent/backend/requirements.txt
```

The backend requirements intentionally do not install `fedot-ind`, `pandas`, `scikit-learn`, `xgboost`, or the official `mcp` package; those either come from Fedot.Industrial or are replaced by the local MCP tool adapter. This keeps the Poetry environment on Pydantic v1, as required by Fedot.Industrial's `spacy` dependency.

## 4. Run Backend

```powershell
cd D:\Diploma\Fedot.Industrial
$env:FEDOT_INDUSTRIAL_PATH="D:\Diploma\Fedot.Industrial"
$env:OPENROUTER_API_KEY="your_key"
poetry run python ..\industrial-learning-agent\backend\app.py
```

Backend URL:

```text
http://localhost:8001
```

Linux:

```bash
cd ~/Fedot.Industrial
export FEDOT_INDUSTRIAL_PATH="$HOME/Fedot.Industrial"
export OPENROUTER_API_KEY="your_key"
poetry run python ../industrial-learning-agent/backend/app.py
```

## 5. Run Frontend

Open a second terminal:

```powershell
cd D:\Diploma\industrial-learning-agent\frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:BACKEND_URL="http://localhost:8001"
streamlit run streamlit_app.py --server.port 8502
```

Frontend URL:

```text
http://localhost:8502
```

## Docker Variant

Docker build expects `Fedot.Industrial` next to this project. The compose file uses:

```yaml
additional_contexts:
  fedot_industrial: ${FEDOT_INDUSTRIAL_CONTEXT:-../Fedot.Industrial}
```

If the checkout is elsewhere:

```powershell
$env:FEDOT_INDUSTRIAL_CONTEXT="D:\path\to\Fedot.Industrial"
docker compose up --build
```

The backend image installs Fedot.Industrial from that local checkout with Poetry.
