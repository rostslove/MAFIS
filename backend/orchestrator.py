import json
import logging
import os
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from agents import Architect, Engineer, Critic, Scribe, DataContext, FedotConfig, CriticFeedback
from agents.schemas import is_ts_task, ALL_PROBLEM_TYPES, IterationRecord
from data_profiler import DataProfiler
from mcp_client import MCPToolClient

logger = logging.getLogger("Orchestrator")


async def run_orchestration_stream(
    csv_path: str,
    target_column: str,
    task_type: str = "classification",
    fedot_url: str = "http://fedot-server:8000",
    iterations: int = 2,
    initial_fedot_config: Optional[Dict[str, Any]] = None,
    forecast_length: Optional[int] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streaming orchestration — yields progress events as SSE.

    Event types:
      - status: general status message
      - agent_start: agent begins work
      - agent_done: agent finished with summary
      - tool_call: MCP tool was called
      - iteration_done: iteration completed with results
      - error: something failed
      - complete: orchestration finished, final result attached
    """

    def event(event_type: str, **data) -> Dict[str, Any]:
        return {"event": event_type, **data}

    # ============== VALIDATE ==============
    if task_type not in ALL_PROBLEM_TYPES:
        yield event("error", message=f"Unknown task type: {task_type}")
        return
    if task_type == "ts_forecasting" and not forecast_length:
        yield event("error", message="ts_forecasting requires forecast_length")
        return

    yield event("status", message="Loading data...")

    # ============== LOAD DATA ==============
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        yield event("error", message=f"Failed to load CSV: {e}")
        return

    if target_column not in df.columns:
        yield event("error", message=f"Target '{target_column}' not found")
        return

    y = df[target_column]
    X = df.drop(columns=[target_column])

    if task_type in ("classification", "ts_classification") and y.dtype == "object":
        y = pd.Series(LabelEncoder().fit_transform(y), name=target_column)

    if task_type == "ts_forecasting":
        idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:idx], X.iloc[idx:]
        y_train, y_val = y.iloc[:idx], y.iloc[idx:]
    else:
        stratify = y if task_type in ("classification", "ts_classification") else None
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

    yield event("status", message=f"Data loaded: {len(X_train)} train, {len(X_val)} val")

    # ============== PROFILE ==============
    profile = DataProfiler.profile(X=X_train, y=y_train, task_type=task_type)
    profile["is_time_series"] = is_ts_task(task_type)
    if forecast_length:
        profile["forecast_length"] = forecast_length

    yield event("status", message=f"Data profiled. Issues: {profile.get('issues', [])}")

    data_context = DataContext(
        csv_path=csv_path, target_column=target_column, task_type=task_type,
        profile=profile, n_train_samples=len(X_train), n_val_samples=len(X_val),
        forecast_length=forecast_length, is_time_series=is_ts_task(task_type),
    )

    # ============== MCP CLIENT ==============
    mcp_client = MCPToolClient()
    try:
        server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
        await mcp_client.connect(server_script=server_script, fedot_url=fedot_url)
        yield event("status", message="MCP server connected")
    except Exception as e:
        yield event("error", message=f"MCP connection failed: {e}")
        return

    try:
        architect = Architect(name="Architect", mcp_client=mcp_client)
        engineer = Engineer(name="Engineer", mcp_client=mcp_client)
        critic = Critic(name="Critic", mcp_client=mcp_client)
        scribe = Scribe(name="Scribe", mcp_client=mcp_client)

        all_results = []
        prev_feedback: Optional[CriticFeedback] = None

        # ============== ITERATION LOOP ==============
        for iteration in range(1, iterations + 1):
            yield event("status", message=f"Iteration {iteration}/{iterations}")

            try:
                # 1. ARCHITECT
                yield event("agent_start", agent="Architect", iteration=iteration, step="1/3")
                baselines, architect_result = await architect.execute(data_context, iteration, prev_feedback, task_type)

                # Force-merge Critic feedback
                if prev_feedback and prev_feedback.suggested_fedot_changes and architect_result.fedot_config:
                    for key, value in prev_feedback.suggested_fedot_changes.items():
                        if hasattr(architect_result.fedot_config, key):
                            setattr(architect_result.fedot_config, key, value)

                fc = architect_result.fedot_config
                yield event("agent_done", agent="Architect", iteration=iteration,
                            summary=f"Baselines: {architect_result.selected_baselines}, Fedot preset: {fc.preset if fc else 'N/A'}",
                            tool_calls_count=len(architect_result.tool_calls))

                # 2. ENGINEER
                yield event("agent_start", agent="Engineer", iteration=iteration, step="2/3")
                engineer_result = await engineer.execute(architect_result, data_context)

                baseline_summary = ", ".join(
                    f"{r.name}={r.score:.3f}" for r in engineer_result.baseline_results if r.score > 0
                )
                fedot_score = engineer_result.fedot_result.get("score", 0)
                yield event("agent_done", agent="Engineer", iteration=iteration,
                            summary=f"Baselines: [{baseline_summary}], Fedot: {fedot_score:.4f}, Best: {engineer_result.best_model}",
                            tool_calls_count=len(engineer_result.tool_calls))

                # 3. CRITIC
                yield event("agent_start", agent="Critic", iteration=iteration, step="3/3")
                critic_result = await critic.execute(engineer_result, data_context, iteration)

                yield event("agent_done", agent="Critic", iteration=iteration,
                            summary=f"Winner: {critic_result.winner}, Stop: {critic_result.should_stop}, Changes: {critic_result.suggested_fedot_changes}",
                            tool_calls_count=len(critic_result.tool_calls))

                # Store results
                iter_data = {
                    "iteration": iteration,
                    "architect": architect_result.to_dict(),
                    "engineer": engineer_result.to_dict(),
                    "critic": critic_result.to_dict(),
                }
                all_results.append(iter_data)

                # Record for history
                failed_bl = [r.name for r in engineer_result.baseline_results if r.error]
                data_context.iteration_history.append(IterationRecord(
                    iteration=iteration,
                    best_model=engineer_result.best_model,
                    best_score=engineer_result.best_score,
                    fedot_score=engineer_result.fedot_result.get("score", 0),
                    fedot_config_used=engineer_result.fedot_config_used.to_dict() if engineer_result.fedot_config_used else {},
                    winner=critic_result.winner,
                    suggested_changes=critic_result.suggested_fedot_changes,
                    failed_baselines=failed_bl,
                ))

                yield event("iteration_done", iteration=iteration,
                            best_score=engineer_result.best_score,
                            best_model=engineer_result.best_model,
                            winner=critic_result.winner)

                prev_feedback = critic_result

                # Stop criteria
                if critic_result.should_stop:
                    yield event("status", message=f"Early stop: Critic recommends stopping")
                    break

                history = data_context.iteration_history
                if len(history) >= 2:
                    improvement = history[-1].best_score - history[-2].best_score
                    if improvement < 0.001 and history[-1].best_score > 0:
                        yield event("status", message=f"Early stop: No improvement (delta={improvement:.4f})")
                        break

            except Exception as e:
                logger.error(f"Iteration {iteration} failed: {e}", exc_info=True)
                all_results.append({"iteration": iteration, "error": str(e), "status": "failed"})
                yield event("error", message=f"Iteration {iteration} failed: {str(e)[:200]}")

        # ============== SCRIBE ==============
        yield event("agent_start", agent="Scribe", iteration=0, step="final")
        try:
            scribe_result = await scribe.execute(all_results, data_context)
            report = scribe_result.to_dict()
        except Exception as e:
            report = {"title": "Report failed", "error": str(e)}
        yield event("agent_done", agent="Scribe", iteration=0, summary=f"Report: {report.get('title', 'N/A')}")

        mcp_tools = await mcp_client.list_tools()

        # Final result
        final_result = {
            "status": "success",
            "task_type": task_type,
            "is_time_series": is_ts_task(task_type),
            "forecast_length": forecast_length,
            "profile": profile,
            "iterations": all_results,
            "best_iteration": _get_best_iteration(all_results),
            "summary": _create_summary(all_results, task_type),
            "report": report,
            "mcp_tools": mcp_tools,
        }

        yield event("complete", result=final_result)

    finally:
        await mcp_client.cleanup()


# Keep non-streaming version for backward compatibility
async def run_orchestration(
    csv_path: str,
    target_column: str,
    task_type: str = "classification",
    fedot_url: str = "http://fedot-server:8000",
    iterations: int = 2,
    initial_fedot_config: Optional[Dict[str, Any]] = None,
    forecast_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Non-streaming wrapper — collects all events and returns final result."""
    final_result = {"error": "No result", "status": "failed"}
    async for evt in run_orchestration_stream(
        csv_path, target_column, task_type, fedot_url, iterations,
        initial_fedot_config, forecast_length,
    ):
        if evt.get("event") == "complete":
            final_result = evt.get("result", final_result)
        elif evt.get("event") == "error" and "result" not in final_result:
            final_result = {"error": evt.get("message", "Unknown error"), "status": "failed"}
    return final_result


def _get_best_iteration(all_results):
    best, best_score = None, -1
    for r in all_results:
        if "error" not in r:
            score = r.get("engineer", {}).get("best_score", -1)
            if score > best_score:
                best_score, best = score, r
    return best or (all_results[-1] if all_results else {})


def _create_summary(all_results, task_type):
    ok = [r for r in all_results if "error" not in r]
    if not ok:
        return {"status": "no successful iterations"}
    a_scores = [r.get("engineer", {}).get("best_score", 0) for r in ok]
    f_scores = [r.get("engineer", {}).get("fedot_result", {}).get("score", 0) for r in ok]
    return {
        "total_iterations": len(all_results),
        "successful_iterations": len(ok),
        "task_type": task_type,
        "is_time_series": is_ts_task(task_type),
        "avg_architect_score": sum(a_scores) / len(a_scores),
        "avg_fedot_score": sum(f_scores) / len(f_scores),
        "max_architect_score": max(a_scores),
        "max_fedot_score": max(f_scores),
        "architect_wins_count": sum(1 for r in ok if r.get("engineer", {}).get("comparison", {}).get("architect_wins", False)),
        "early_stopped": any(r.get("critic", {}).get("should_stop", False) for r in ok),
    }
