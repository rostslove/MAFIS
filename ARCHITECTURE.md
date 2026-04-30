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
        | MCP stdio subprocess
        v
MCP graph tools
        |
        | direct Python calls
        v
Fedot.Industrial + sklearn
```

There is no separate `fedot_server` service anymore. The old double serialization path (`MCP -> HTTP -> Fedot server`) was removed; MCP tools now call graph and training code directly in the backend container.

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

## API Endpoints

- `GET /health`
- `GET /config`
- `GET /tools`
- `POST /architect/chat`
- `POST /graph/mutate`
- `POST /orchestrate`
- `POST /orchestrate/stream`

`/orchestrate/file` was removed because the frontend writes uploaded CSVs into the shared `data` volume and calls `/orchestrate/stream`.
