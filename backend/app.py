from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import json
import logging
import tempfile
import os
from typing import Optional, Dict, Any

from orchestrator import run_orchestration, run_orchestration_stream
from agents.base_agent import LLM_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Multi-Agent System (MCP)", version="3.0.0")

# Agent → Tool mapping
AGENT_TOOLS = {
    "architect": [
        "get_data_profile", "get_available_baselines",
        "get_available_operations", "propose_fedot_config", "mutate_fedot_config",
    ],
    "engineer": ["train_baseline", "call_fedot", "compare_results"],
    "critic": ["analyze_errors", "get_feature_importance", "explain_model", "get_additional_metrics"],
    "scribe": ["generate_report"],
}

TOOL_DESCRIPTIONS = {
    "get_data_profile": "Profile dataset: statistics, issues, recommendations",
    "get_available_baselines": "Get sklearn baselines for classification/regression",
    "get_available_operations": "Get Fedot.Industrial models/preprocessing/strategies for any task type",
    "propose_fedot_config": "Propose Fedot.Industrial configuration (preset, strategy, operations...)",
    "mutate_fedot_config": "Modify config based on Critic feedback",
    "train_baseline": "Train a sklearn pipeline and get metrics",
    "call_fedot": "Call Fedot.Industrial with full parameter control (all task types)",
    "compare_results": "Compare all models and determine the best",
    "analyze_errors": "Analyze prediction errors, baseline vs Fedot comparison",
    "get_feature_importance": "Feature importance from tree-based models",
    "explain_model": "Explain Fedot model predictions (point/recurrence/shap/lime)",
    "get_additional_metrics": "Calculate additional metrics from Fedot",
    "generate_report": "Compile iteration results into structured report",
}


class OrchestrationRequest(BaseModel):
    csv_path: str
    target_column: str
    task_type: str = "classification"
    fedot_url: str = "http://fedot-server:8000"
    iterations: int = 3
    initial_fedot_config: Optional[Dict[str, Any]] = None
    forecast_length: Optional[int] = None


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "ml-multi-agent-mcp"}


@app.get("/config")
async def get_config():
    return {
        "agents": ["Architect", "Engineer", "Critic", "Scribe"],
        "llm_model": LLM_MODEL,
        "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "fedot_host": os.getenv("FEDOT_HOST", "http://fedot-server:8000"),
        "protocol": "MCP",
        "transport": "stdio",
    }


@app.get("/tools")
async def get_tools():
    """Return all MCP tools grouped by agent."""
    result = {}
    for agent, tool_names in AGENT_TOOLS.items():
        result[agent] = [
            {"name": name, "description": TOOL_DESCRIPTIONS.get(name, "")}
            for name in tool_names
        ]
    return result


@app.post("/orchestrate")
async def orchestrate(request: OrchestrationRequest):
    """Non-streaming orchestration (returns full result at end)."""
    try:
        result = await run_orchestration(
            csv_path=request.csv_path,
            target_column=request.target_column,
            task_type=request.task_type,
            fedot_url=request.fedot_url,
            iterations=request.iterations,
            initial_fedot_config=request.initial_fedot_config,
            forecast_length=request.forecast_length,
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/orchestrate/stream")
async def orchestrate_stream(request: OrchestrationRequest):
    """SSE streaming orchestration — sends progress events in real time."""

    async def event_generator():
        try:
            async for evt in run_orchestration_stream(
                csv_path=request.csv_path,
                target_column=request.target_column,
                task_type=request.task_type,
                fedot_url=request.fedot_url,
                iterations=request.iterations,
                initial_fedot_config=request.initial_fedot_config,
                forecast_length=request.forecast_length,
            ):
                data = json.dumps(evt, default=str, ensure_ascii=False)
                yield f"data: {data}\n\n"
        except Exception as e:
            error_data = json.dumps({"event": "error", "message": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/orchestrate/file")
async def orchestrate_file(
    file: UploadFile = File(...),
    target_column: str = None,
    task_type: str = "classification",
    iterations: int = 3,
    forecast_length: int = None,
):
    if not target_column:
        raise HTTPException(status_code=400, detail="target_column required")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            result = await run_orchestration(
                csv_path=tmp_path, target_column=target_column, task_type=task_type,
                fedot_url=os.getenv("FEDOT_HOST", "http://fedot-server:8000"),
                iterations=iterations, forecast_length=forecast_length,
            )
            return JSONResponse(content=result)
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
