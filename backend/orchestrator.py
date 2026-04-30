import logging
import os
from typing import Any, AsyncGenerator, Dict, Optional

import pandas as pd

from agents import Architect, ArchitectResult, Critic, CriticFeedback, DataContext, Engineer, IterationRecord, Scribe
from data_profiler import DataProfiler
from graph_engine import SUPPORTED_TASKS, PipelineGraph, is_ts_task
from mcp_client import MCPToolClient

logger = logging.getLogger("Orchestrator")


def _event(event_type: str, **data) -> Dict[str, Any]:
    return {"event": event_type, **data}


def _profile_data(csv_path: str, target_column: str, task_type: str, forecast_length: Optional[int]) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' not found")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    profile = DataProfiler.profile(X=X, y=y, task_type=task_type)
    profile["is_time_series"] = is_ts_task(task_type)
    if forecast_length:
        profile["forecast_length"] = forecast_length
    return profile


async def _connect_mcp() -> MCPToolClient:
    client = MCPToolClient()
    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
    await client.connect(server_script=server_script)
    return client


async def run_orchestration_stream(
    csv_path: str,
    target_column: str,
    task_type: str = "classification",
    fedot_url: str = "",
    iterations: int = 2,
    initial_graph: Optional[Dict[str, Any]] = None,
    initial_fedot_config: Optional[Dict[str, Any]] = None,
    forecast_length: Optional[int] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Run Architect -> Engineer -> Critic iterations and yield SSE-friendly events."""
    del fedot_url, initial_fedot_config

    if task_type not in SUPPORTED_TASKS:
        yield _event("error", message=f"Unknown task type: {task_type}. Available: {SUPPORTED_TASKS}")
        return
    if task_type == "ts_forecasting" and not forecast_length:
        yield _event("error", message="ts_forecasting requires forecast_length")
        return

    try:
        yield _event("status", message="Loading and profiling data...")
        profile = _profile_data(csv_path, target_column, task_type, forecast_length)
    except Exception as exc:
        yield _event("error", message=f"Failed to load data: {exc}")
        return

    data_context = DataContext(
        csv_path=csv_path,
        target_column=target_column,
        task_type=task_type,
        profile=profile,
        forecast_length=forecast_length,
    )
    yield _event(
        "status",
        message=f"Data profiled: {profile.get('n_samples')} samples, {profile.get('n_features')} numeric features",
    )

    mcp_client: Optional[MCPToolClient] = None
    try:
        mcp_client = await _connect_mcp()
        yield _event("status", message="MCP graph tools connected")

        architect = Architect(mcp_client=mcp_client)
        engineer = Engineer(mcp_client=mcp_client)
        critic = Critic(mcp_client=mcp_client)
        scribe = Scribe(mcp_client=mcp_client)

        all_results = []
        prev_feedback: Optional[CriticFeedback] = None
        prev_graph = initial_graph
        approved_initial: Optional[ArchitectResult] = None

        if initial_graph:
            graph = PipelineGraph.from_dict(initial_graph)
            ok, message = graph.validate()
            if not ok:
                yield _event("error", message=f"Initial graph is invalid: {message}")
                return
            approved_initial = ArchitectResult(
                graph=graph.to_dict(),
                mermaid=graph.to_mermaid(),
                analysis="User-approved initial graph",
                reasoning="The first iteration trains the approved graph without structural changes.",
            )

        for iteration in range(1, max(1, iterations) + 1):
            yield _event("status", message=f"Iteration {iteration}/{iterations}")

            try:
                yield _event("agent_start", agent="Architect", iteration=iteration, step="1/3")
                if iteration == 1 and approved_initial:
                    architect_result = approved_initial
                else:
                    architect_result = await architect.execute(
                        data_context=data_context,
                        iteration=iteration,
                        prev_feedback=prev_feedback,
                        prev_graph=prev_graph,
                    )
                yield _event(
                    "agent_done",
                    agent="Architect",
                    iteration=iteration,
                    summary=f"Graph nodes: {len(architect_result.graph.get('nodes', []))}",
                    graph=architect_result.graph,
                    mermaid=architect_result.mermaid,
                    tool_calls_count=len(architect_result.tool_calls),
                )

                yield _event("agent_start", agent="Engineer", iteration=iteration, step="2/3")
                engineer_result = await engineer.execute(architect_result, data_context)
                yield _event(
                    "agent_done",
                    agent="Engineer",
                    iteration=iteration,
                    summary=(
                        f"Graph score: {engineer_result.graph_score:.4f}; "
                        f"best baseline: {engineer_result.best_baseline_name or 'none'} "
                        f"{engineer_result.best_baseline_score:.4f}"
                    ),
                    tool_calls_count=len(engineer_result.tool_calls),
                )

                yield _event("agent_start", agent="Critic", iteration=iteration, step="3/3")
                critic_result = await critic.execute(architect_result, engineer_result, data_context, iteration)
                yield _event(
                    "agent_done",
                    agent="Critic",
                    iteration=iteration,
                    summary=(
                        f"Winner: {critic_result.winner}; stop={critic_result.should_stop}; "
                        f"mutations={len(critic_result.suggested_mutations)}"
                    ),
                    tool_calls_count=len(critic_result.tool_calls),
                )

                iter_data = {
                    "iteration": iteration,
                    "architect": architect_result.to_dict(),
                    "engineer": engineer_result.to_dict(),
                    "critic": critic_result.to_dict(),
                }
                all_results.append(iter_data)
                data_context.iteration_history.append(
                    IterationRecord(
                        iteration=iteration,
                        graph=architect_result.graph,
                        graph_score=engineer_result.graph_score,
                        best_baseline_score=engineer_result.best_baseline_score,
                        winner=critic_result.winner,
                        suggested_mutations=critic_result.suggested_mutations,
                    )
                )

                yield _event(
                    "iteration_done",
                    iteration=iteration,
                    graph_score=engineer_result.graph_score,
                    best_baseline_score=engineer_result.best_baseline_score,
                    winner=critic_result.winner,
                    graph=architect_result.graph,
                    mermaid=architect_result.mermaid,
                )

                prev_feedback = critic_result
                prev_graph = architect_result.graph

                if critic_result.should_stop:
                    yield _event("status", message="Early stop: Critic accepted the graph")
                    break

                if len(data_context.iteration_history) >= 2:
                    last = data_context.iteration_history[-1]
                    before = data_context.iteration_history[-2]
                    if abs(last.graph_score - before.graph_score) < 0.001 and last.graph_score > 0:
                        yield _event("status", message="Early stop: graph score plateaued")
                        break

            except Exception as exc:
                logger.exception("Iteration %s failed", iteration)
                all_results.append({"iteration": iteration, "error": str(exc), "status": "failed"})
                yield _event("error", message=f"Iteration {iteration} failed: {str(exc)[:200]}")

        yield _event("agent_start", agent="Scribe", iteration=0, step="final")
        scribe_result = await scribe.execute(all_results, data_context)
        yield _event(
            "agent_done",
            agent="Scribe",
            iteration=0,
            summary=f"Report: {scribe_result.title}",
            tool_calls_count=len(scribe_result.tool_calls),
        )

        mcp_tools = await mcp_client.list_tools()
        final_result = {
            "status": "success",
            "task_type": task_type,
            "is_time_series": is_ts_task(task_type),
            "forecast_length": forecast_length,
            "profile": profile,
            "iterations": all_results,
            "best_iteration": _get_best_iteration(all_results),
            "summary": _create_summary(all_results, task_type),
            "report": scribe_result.to_dict(),
            "mcp_tools": mcp_tools,
        }
        yield _event("complete", result=final_result)

    finally:
        if mcp_client:
            await mcp_client.cleanup()


async def run_orchestration(
    csv_path: str,
    target_column: str,
    task_type: str = "classification",
    fedot_url: str = "",
    iterations: int = 2,
    initial_graph: Optional[Dict[str, Any]] = None,
    initial_fedot_config: Optional[Dict[str, Any]] = None,
    forecast_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Collect streaming events and return the final result."""
    final_result: Dict[str, Any] = {"status": "failed", "error": "No result"}
    async for evt in run_orchestration_stream(
        csv_path=csv_path,
        target_column=target_column,
        task_type=task_type,
        fedot_url=fedot_url,
        iterations=iterations,
        initial_graph=initial_graph,
        initial_fedot_config=initial_fedot_config,
        forecast_length=forecast_length,
    ):
        if evt.get("event") == "complete":
            final_result = evt.get("result", final_result)
        elif evt.get("event") == "error" and final_result.get("status") == "failed":
            final_result = {"status": "failed", "error": evt.get("message", "Unknown error")}
    return final_result


async def propose_architecture(
    csv_path: str,
    target_column: str,
    task_type: str,
    message: str = "",
    current_graph: Optional[Dict[str, Any]] = None,
    forecast_length: Optional[int] = None,
) -> Dict[str, Any]:
    """One-shot Architect interaction for the frontend graph approval flow."""
    if task_type not in SUPPORTED_TASKS:
        return {"error": f"Unknown task type: {task_type}"}

    profile = _profile_data(csv_path, target_column, task_type, forecast_length)
    data_context = DataContext(
        csv_path=csv_path,
        target_column=target_column,
        task_type=task_type,
        profile=profile,
        forecast_length=forecast_length,
    )

    mcp_client = await _connect_mcp()
    try:
        architect = Architect(mcp_client=mcp_client)
        feedback = None
        if message:
            feedback = CriticFeedback(
                winner="user",
                weaknesses=[message],
                suggested_mutations=[],
            )
        result = await architect.execute(
            data_context=data_context,
            iteration=1,
            prev_feedback=feedback,
            prev_graph=current_graph,
        )
        return {
            "profile": profile,
            "graph": result.graph,
            "mermaid": result.mermaid,
            "analysis": result.analysis,
            "reasoning": result.reasoning,
            "tool_calls": [tc.to_dict() for tc in result.tool_calls],
        }
    finally:
        await mcp_client.cleanup()


def mutate_graph_locally(graph: Dict[str, Any], mutation: Dict[str, Any]) -> Dict[str, Any]:
    pipeline_graph = PipelineGraph.from_dict(graph)
    mutated = pipeline_graph.apply_mutation(mutation)
    ok, message = mutated.validate()
    return {
        "valid": ok,
        "message": message,
        "graph": mutated.to_dict(),
        "mermaid": mutated.to_mermaid() if ok else "",
    }


def _get_best_iteration(all_results):
    best, best_score = None, -1.0
    for item in all_results:
        if "error" in item:
            continue
        score = float(item.get("engineer", {}).get("graph_score", 0) or 0)
        if score > best_score:
            best_score = score
            best = item
    return best or (all_results[-1] if all_results else {})


def _create_summary(all_results, task_type):
    ok = [item for item in all_results if "error" not in item]
    if not ok:
        return {"status": "no successful iterations", "task_type": task_type}

    graph_scores = [float(item.get("engineer", {}).get("graph_score", 0) or 0) for item in ok]
    baseline_scores = [float(item.get("engineer", {}).get("best_baseline_score", 0) or 0) for item in ok]
    return {
        "total_iterations": len(all_results),
        "successful_iterations": len(ok),
        "task_type": task_type,
        "is_time_series": is_ts_task(task_type),
        "avg_graph_score": sum(graph_scores) / len(graph_scores),
        "avg_baseline_score": sum(baseline_scores) / len(baseline_scores),
        "max_graph_score": max(graph_scores),
        "max_baseline_score": max(baseline_scores),
        "graph_wins_count": sum(
            1 for item in ok if item.get("critic", {}).get("winner") == "graph"
        ),
        "early_stopped": any(item.get("critic", {}).get("should_stop", False) for item in ok),
    }
