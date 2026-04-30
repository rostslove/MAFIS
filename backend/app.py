import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from agents.base_agent import LLM_MODEL
from graph_engine import METRICS_BY_TASK, OPERATIONS, SUPPORTED_TASKS
from orchestrator import mutate_graph_locally, propose_architecture, run_orchestration, run_orchestration_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GraphAutoML MCP System", version="4.0.0")


AGENT_TOOLS = {
    "architect": ["get_data_profile", "get_available_operations", "propose_graph", "mutate_graph", "visualize_graph"],
    "engineer": ["get_baselines", "train_baseline", "tune_graph_hyperparameters", "train_graph"],
    "critic": ["validate_graph", "analyze_errors", "get_node_importance", "explain_graph"],
    "scribe": ["generate_report", "visualize_graph"],
}

TOOL_DESCRIPTIONS = {
    "get_data_profile": "Profile dataset shape, target distribution, issues and recommendations",
    "get_available_operations": "Return atomic graph operations allowed for a task type",
    "propose_graph": "Validate a graph candidate and return Mermaid markup",
    "mutate_graph": "Apply add/remove/replace/set_params/connect mutations to a graph",
    "visualize_graph": "Render graph JSON to Mermaid markup",
    "get_baselines": "List simple sklearn baselines for the task",
    "train_baseline": "Train and score one baseline model",
    "tune_graph_hyperparameters": "Tune node hyperparameters without changing graph structure",
    "train_graph": "Fit and score a graph exactly as proposed",
    "validate_graph": "Cross-validate a graph",
    "analyze_errors": "Compare graph score with baseline scores",
    "get_node_importance": "Estimate node contribution by ablation",
    "explain_graph": "Explain the last trained graph if model internals expose importances",
    "generate_report": "Compile iteration results into a final report summary",
}


class OrchestrationRequest(BaseModel):
    csv_path: str
    target_column: str
    task_type: str = "classification"
    iterations: int = 3
    initial_graph: Optional[Dict[str, Any]] = None
    forecast_length: Optional[int] = None


class ArchitectChatRequest(BaseModel):
    csv_path: str
    target_column: str
    task_type: str = "classification"
    message: str = ""
    current_graph: Optional[Dict[str, Any]] = None
    forecast_length: Optional[int] = None


class GraphMutationRequest(BaseModel):
    graph: Dict[str, Any]
    mutation: Dict[str, Any]


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "graph-automl-mcp"}


@app.get("/config")
async def get_config():
    return {
        "agents": ["Architect", "Engineer", "Critic", "Scribe"],
        "llm_model": LLM_MODEL,
        "protocol": "MCP",
        "transport": "stdio",
        "supported_tasks": SUPPORTED_TASKS,
        "metrics_by_task": METRICS_BY_TASK,
        "operations": OPERATIONS,
        "openrouter_configured": bool(os.getenv("OPENROUTER_API_KEY")),
    }


@app.get("/tools")
async def get_tools():
    return {
        agent: [
            {"name": name, "description": TOOL_DESCRIPTIONS.get(name, "")}
            for name in names
        ]
        for agent, names in AGENT_TOOLS.items()
    }


@app.post("/architect/chat")
async def architect_chat(request: ArchitectChatRequest):
    try:
        result = await propose_architecture(
            csv_path=request.csv_path,
            target_column=request.target_column,
            task_type=request.task_type,
            message=request.message,
            current_graph=request.current_graph,
            forecast_length=request.forecast_length,
        )
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Architect chat failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/graph/mutate")
async def graph_mutate(request: GraphMutationRequest):
    try:
        result = mutate_graph_locally(request.graph, request.mutation)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/orchestrate")
async def orchestrate(request: OrchestrationRequest):
    try:
        result = await run_orchestration(
            csv_path=request.csv_path,
            target_column=request.target_column,
            task_type=request.task_type,
            iterations=request.iterations,
            initial_graph=request.initial_graph,
            forecast_length=request.forecast_length,
        )
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("Orchestration failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/orchestrate/stream")
async def orchestrate_stream(request: OrchestrationRequest):
    async def event_generator():
        try:
            async for evt in run_orchestration_stream(
                csv_path=request.csv_path,
                target_column=request.target_column,
                task_type=request.task_type,
                iterations=request.iterations,
                initial_graph=request.initial_graph,
                forecast_length=request.forecast_length,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
