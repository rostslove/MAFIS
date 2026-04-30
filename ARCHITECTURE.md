# GraphAutoML Architecture

GraphAutoML is a multi-agent system where LLM agents compose, tune, validate, and explain Fedot.Industrial pipeline graphs through MCP tools.

The important design choice is that the LLM controls graph structure. Fedot.Industrial is used to materialize and train the graph, and the Engineer may tune node hyperparameters, but structural evolution is performed by the Architect/Critic loop.

## Runtime Layout

```text
Streamlit frontend (:8502)
        |
        | HTTP / SSE
        v
FastAPI backend (:8001)
        |
        | MCP-style local tool adapter
        v
MCP graph tools
        |
        | direct Python calls
        v
Fedot.Industrial + sklearn
```

There is no separate `fedot_server` service anymore. The old double serialization path (`MCP -> HTTP -> Fedot server`) was removed; MCP tools now call graph and training code directly in the backend container.

Fedot.Industrial 0.5.0 is expected to come from a local source checkout, not from PyPI. The backend supports a sibling checkout at `../Fedot.Industrial` or an explicit `FEDOT_INDUSTRIAL_PATH`.

## Agents

### Architect

Input: user task, CSV profile, optional previous graph, optional Critic or user feedback.

Main tools:

- `get_data_profile`
- `get_available_operations`
- `propose_graph`
- `mutate_graph`
- `visualize_graph`

Output: `PipelineGraph` JSON and Mermaid markup.

### Engineer

Input: validated graph from Architect.

Main tools:

- `get_baselines`
- `train_baseline`
- `tune_graph_hyperparameters`
- `train_graph`

Output: graph score, metrics, tuned node parameters, baseline comparison.

Engineer does not change graph structure. It only fits the graph and tunes node parameters.

### Critic

Input: graph, Engineer metrics, baseline metrics.

Main tools:

- `validate_graph`
- `analyze_errors`
- `get_node_importance`
- `explain_graph`

Output: assessment, winner (`graph` or `baseline`), strengths, weaknesses, suggested graph mutations, stop decision.

### Scribe

Input: all iteration records.

Main tools:

- `generate_report`
- `visualize_graph`

Output: final report, best graph visualization.

## Iteration Flow

```text
1. Backend profiles the CSV.
2. Architect creates or mutates a graph candidate.
3. MCP validates the graph shape and allowed operations.
4. Engineer trains simple baselines and the proposed graph.
5. Critic cross-validates, analyzes errors, and suggests mutations.
6. If Critic stops or scores plateau, the loop ends.
7. Otherwise feedback returns to Architect for the next graph.
8. Scribe creates the final report.
```

## Graph Format

```json
{
  "task_type": "classification",
  "nodes": [
    {"id": "scale", "operation": "scaling", "params": {}, "inputs": []},
    {"id": "model", "operation": "rf", "params": {}, "inputs": ["scale"]}
  ]
}
```

Rules:

- `task_type` must be one of `classification`, `regression`, `ts_classification`, `ts_regression`, `ts_forecasting`.
- Each node has a unique `id`.
- `operation` must be allowed for the selected task type.
- `inputs` contain upstream node IDs.
- The graph must be a DAG with exactly one root node.
- The root node is the final model.

## Supported Tasks

Current scope:

- classification
- regression
- ts_classification
- ts_regression
- ts_forecasting

Anomaly detection was removed from the active flow to keep the first graph-oriented version focused.

## Frontend Flow

The Streamlit app is organized around graph approval:

1. Upload CSV and choose target/task.
2. Ask Architect to propose a graph.
3. Inspect available operations and graph visualization.
4. Optionally edit the graph by adding/removing/replacing nodes or setting params.
5. Approve the graph and run iterative GraphAutoML.
6. Review metrics, Critic feedback, and final Scribe report.

## Main Files

- `backend/graph_engine.py` - graph schema, validation, mutation, Fedot conversion, metrics.
- `backend/mcp_server.py` - MCP tools for profiling, graph validation, training, validation, reporting.
- `backend/orchestrator.py` - agent loop and Architect chat helper.
- `backend/agents/` - agent implementations and shared dataclasses.
- `backend/app.py` - FastAPI endpoints.
- `frontend/streamlit_app.py` - interactive graph-first UI.

## Fedot.Industrial 0.5.0 Setup

`fedot-ind==0.5.0` is installed from the repository checkout:

```bash
git clone https://github.com/aimclub/Fedot.Industrial.git
cd Fedot.Industrial
poetry install
```

For local backend launch, run the backend inside this Poetry environment and install only the application dependencies from `backend/requirements.txt`.

The backend requirements deliberately do not pin `pandas`, `scikit-learn`, or `xgboost`; those versions must remain controlled by `Fedot.Industrial/poetry.lock`.

The official Python MCP SDK is not installed in the Fedot.Industrial Poetry environment because current releases require Pydantic v2, while Fedot.Industrial 0.5.0 depends on `spacy 3.5.x`, which requires Pydantic v1. The project keeps an MCP-style tool registry and OpenAI tool schema adapter locally.

For Docker launch, `docker-compose.yml` passes the sibling checkout as a BuildKit additional context:

```yaml
additional_contexts:
  fedot_industrial: ${FEDOT_INDUSTRIAL_CONTEXT:-../Fedot.Industrial}
```

If your checkout has another path or name, set `FEDOT_INDUSTRIAL_CONTEXT` for Docker and `FEDOT_INDUSTRIAL_PATH` for local Python runs.

## API Endpoints

- `GET /health`
- `GET /config`
- `GET /tools`
- `POST /architect/chat`
- `POST /graph/mutate`
- `POST /orchestrate`
- `POST /orchestrate/stream`

`/orchestrate/file` was removed because the frontend writes uploaded CSVs into the shared `data` volume and calls `/orchestrate/stream`.
