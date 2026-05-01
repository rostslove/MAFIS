import json
import logging
import os
from typing import Any, Dict

import anyio
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from agents.base_agent import LLM_BASE_URL, LLM_MODEL
from graph_engine import (
    DEFAULT_GRAPHS,
    FEDOT_IND_VERSION,
    FEDOT_INDUSTRIAL_SOURCE,
    METRICS_BY_TASK,
    OPERATIONS,
    OPERATION_DESCRIPTIONS,
    SUPPORTED_TASKS,
    get_operation_catalog,
    get_training_strategy_hints,
)
from orchestrator import (
    mutate_graph_locally,
    propose_architecture,
    propose_m4_benchmark_architecture,
    propose_revision_from_critic,
    run_orchestration,
    run_orchestration_stream,
    tune_approved_graph,
)
from m4_benchmark import M4_GROUPS, run_m4_classification_benchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


AGENT_TOOLS = {
    "architect": ["get_data_profile", "get_available_operations", "propose_graph", "mutate_graph", "visualize_graph"],
    "engineer": ["tune_graph_hyperparameters", "train_graph"],
    "critic": ["validate_graph", "get_node_importance", "explain_graph"],
    "scribe": ["generate_report"],
}

TOOL_DESCRIPTIONS = {
    "get_data_profile": "Profile dataset shape, target distribution, issues and recommendations",
    "get_available_operations": "Return atomic graph operations allowed for a task type",
    "propose_graph": "Validate a graph candidate and return Mermaid markup",
    "mutate_graph": "Apply add/remove/replace/set_params/connect mutations to a graph",
    "visualize_graph": "Render graph JSON to Mermaid markup",
    "tune_graph_hyperparameters": "Tune node hyperparameters without changing graph structure",
    "train_graph": "Fit and score a graph exactly as proposed",
    "validate_graph": "Cross-validate a graph",
    "get_node_importance": "Estimate node contribution by ablation",
    "explain_graph": "Explain the last trained graph if model internals expose importances",
    "generate_report": "Compile evaluation results into a final report summary",
}


async def _json_body(request) -> Dict[str, Any]:
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"detail": message}, status_code=status_code)


async def health(request):
    return JSONResponse({"status": "healthy", "service": "graph-automl-mcp"})


async def get_config(request):
    llm_provider = "ollama" if "11434" in LLM_BASE_URL or "ollama" in LLM_BASE_URL else "openai-compatible"
    return JSONResponse({
        "agents": ["Architect", "Engineer", "Critic", "Scribe"],
        "llm_provider": llm_provider,
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "protocol": "MCP",
        "transport": "local-adapter",
        "supported_tasks": SUPPORTED_TASKS,
        "metrics_by_task": METRICS_BY_TASK,
        "operations": OPERATIONS,
        "operation_catalog": {task: get_operation_catalog(task) for task in SUPPORTED_TASKS},
        "operation_descriptions": OPERATION_DESCRIPTIONS,
        "training_strategy_hints": {task: get_training_strategy_hints(task) for task in SUPPORTED_TASKS},
        "default_graphs": DEFAULT_GRAPHS,
        "llm_configured": bool(os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or LLM_BASE_URL),
        "openrouter_configured": "openrouter.ai" in LLM_BASE_URL and bool(os.getenv("OPENROUTER_API_KEY")),
        "fedot_ind_version": FEDOT_IND_VERSION,
        "fedot_industrial_source": FEDOT_INDUSTRIAL_SOURCE,
        "benchmarks": {
            "m4_classification": {
                "name": "M4 frequency-group classification",
                "task_type": "ts_classification",
                "requires_architect_graph": True,
                "groups": list(M4_GROUPS),
                "default_primary_metric": "f1",
                "default_window_length": 50,
                "default_n_per_group": 100,
            }
        },
    })


async def get_tools(request):
    return JSONResponse({
        agent: [
            {"name": name, "description": TOOL_DESCRIPTIONS.get(name, "")}
            for name in names
        ]
        for agent, names in AGENT_TOOLS.items()
    })


async def architect_chat(request):
    payload = await _json_body(request)
    try:
        result = await propose_architecture(
            csv_path=payload.get("csv_path", ""),
            target_column=payload.get("target_column", ""),
            task_type=payload.get("task_type", "classification"),
            message=payload.get("message", ""),
            current_graph=payload.get("current_graph"),
            forecast_length=payload.get("forecast_length"),
            primary_metric=payload.get("primary_metric"),
        )
        if result.get("error"):
            return JSONResponse(result, status_code=400)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("Architect chat failed")
        return _error(str(exc), 500)


async def architect_revise(request):
    payload = await _json_body(request)
    try:
        result = await propose_revision_from_critic(
            csv_path=payload.get("csv_path", ""),
            target_column=payload.get("target_column", ""),
            task_type=payload.get("task_type", "classification"),
            current_graph=payload.get("current_graph") or {},
            critic_feedback=payload.get("critic_feedback") or {},
            message=payload.get("message", ""),
            forecast_length=payload.get("forecast_length"),
            primary_metric=payload.get("primary_metric"),
            selected_mutations=payload.get("selected_mutations"),
        )
        if result.get("error"):
            return JSONResponse(result, status_code=400)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("Architect revision failed")
        return _error(str(exc), 500)


async def graph_mutate(request):
    payload = await _json_body(request)
    try:
        result = mutate_graph_locally(payload.get("graph", {}), payload.get("mutation", {}))
        return JSONResponse(result)
    except Exception as exc:
        return _error(str(exc), 400)


def _payload_test_size(payload: Dict[str, Any]) -> float:
    try:
        ts = float(payload.get("test_size", 0.2))
    except (TypeError, ValueError):
        ts = 0.2
    return min(max(ts, 0.05), 0.5)


async def orchestrate(request):
    payload = await _json_body(request)
    try:
        result = await run_orchestration(
            csv_path=payload.get("csv_path", ""),
            target_column=payload.get("target_column", ""),
            task_type=payload.get("task_type", "classification"),
            iterations=int(payload.get("iterations", 1) or 1),
            initial_graph=payload.get("initial_graph"),
            forecast_length=payload.get("forecast_length"),
            primary_metric=payload.get("primary_metric"),
            test_size=_payload_test_size(payload),
        )
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("Orchestration failed")
        return _error(str(exc), 500)


async def engineer_tune(request):
    payload = await _json_body(request)
    try:
        result = await tune_approved_graph(
            csv_path=payload.get("csv_path", ""),
            target_column=payload.get("target_column", ""),
            task_type=payload.get("task_type", "classification"),
            graph=payload.get("graph") or {},
            forecast_length=payload.get("forecast_length"),
            primary_metric=payload.get("primary_metric"),
            test_size=_payload_test_size(payload),
            iterations=int(payload.get("iterations", 30) or 30),
        )
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("Engineer tuning failed")
        return _error(str(exc), 500)


async def benchmark_m4(request):
    payload = await _json_body(request)
    try:
        result = await anyio.to_thread.run_sync(
            lambda: run_m4_classification_benchmark(
                graph=payload.get("graph"),
                n_per_group=int(payload.get("n_per_group", 100) or 100),
                window_length=int(payload.get("window_length", 50) or 50),
                test_size=_payload_test_size(payload),
                primary_metric=payload.get("primary_metric") or "f1",
                random_state=int(payload.get("random_state", 42) or 42),
                standardize=bool(payload.get("standardize", True)),
                groups=payload.get("groups") or None,
            )
        )
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("M4 benchmark failed")
        return _error(str(exc), 500)


async def benchmark_m4_architect(request):
    payload = await _json_body(request)
    try:
        result = await propose_m4_benchmark_architecture(
            message=payload.get("message", ""),
            current_graph=payload.get("current_graph"),
            n_per_group=int(payload.get("n_per_group", 100) or 100),
            window_length=int(payload.get("window_length", 50) or 50),
            test_size=_payload_test_size(payload),
            primary_metric=payload.get("primary_metric") or "f1",
            groups=payload.get("groups") or None,
            standardize=bool(payload.get("standardize", True)),
        )
        if result.get("error"):
            return JSONResponse(result, status_code=400)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("M4 benchmark architect failed")
        return _error(str(exc), 500)


async def orchestrate_stream(request):
    payload = await _json_body(request)

    async def event_generator():
        try:
            async for evt in run_orchestration_stream(
                csv_path=payload.get("csv_path", ""),
                target_column=payload.get("target_column", ""),
                task_type=payload.get("task_type", "classification"),
                iterations=int(payload.get("iterations", 1) or 1),
                initial_graph=payload.get("initial_graph"),
                forecast_length=payload.get("forecast_length"),
                primary_metric=payload.get("primary_metric"),
                test_size=_payload_test_size(payload),
            ):
                yield f"data: {json.dumps(evt, default=str, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("Streaming orchestration failed")
            error = {"event": "error", "message": str(exc)}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


routes = [
    Route("/health", health, methods=["GET"]),
    Route("/config", get_config, methods=["GET"]),
    Route("/tools", get_tools, methods=["GET"]),
    Route("/architect/chat", architect_chat, methods=["POST"]),
    Route("/architect/revise", architect_revise, methods=["POST"]),
    Route("/graph/mutate", graph_mutate, methods=["POST"]),
    Route("/orchestrate", orchestrate, methods=["POST"]),
    Route("/orchestrate/stream", orchestrate_stream, methods=["POST"]),
    Route("/engineer/tune", engineer_tune, methods=["POST"]),
    Route("/benchmarks/m4/architect", benchmark_m4_architect, methods=["POST"]),
    Route("/benchmarks/m4", benchmark_m4, methods=["POST"]),
]

app = Starlette(debug=False, routes=routes)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
