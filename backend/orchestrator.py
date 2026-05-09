import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import pandas as pd

from agents import Architect, ArchitectResult, Critic, CriticFeedback, DataContext, Engineer, EngineerResult, IterationRecord, Scribe
from data_profiler import DataProfiler
from graph_engine import SUPPORTED_TASKS, PipelineGraph, is_industrial_native_model, is_ts_task
from mcp_client import MCPToolClient
from path_utils import describe_missing_csv, normalize_csv_path

logger = logging.getLogger("Orchestrator")


def _event(event_type: str, **data) -> Dict[str, Any]:
    return {"event": event_type, **data}


def _profile_data(csv_path: str, target_column: str, task_type: str, forecast_length: Optional[int]) -> Dict[str, Any]:
    csv_path = normalize_csv_path(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(describe_missing_csv(csv_path))
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


def _execute_engineer_sync(
    engineer: Engineer,
    architect_result: ArchitectResult,
    data_context: DataContext,
) -> EngineerResult:
    """Run long local MCP training away from the SSE event loop."""
    return asyncio.run(engineer.execute(architect_result, data_context))


def _graph_has_industrial_native_model(graph: Dict[str, Any]) -> bool:
    return any(
        is_industrial_native_model(str(node.get("operation") or ""))
        for node in graph.get("nodes", [])
        if isinstance(node, dict)
    )


async def run_orchestration_stream(
    csv_path: str,
    target_column: str,
    task_type: str = "classification",
    fedot_url: str = "",
    iterations: int = 2,
    initial_graph: Optional[Dict[str, Any]] = None,
    initial_fedot_config: Optional[Dict[str, Any]] = None,
    forecast_length: Optional[int] = None,
    primary_metric: Optional[str] = None,
    test_size: float = 0.2,
    industrial_strategy: str = "default",
    industrial_strategy_params: Optional[Dict[str, Any]] = None,
    previous_evaluations: Optional[List[Dict[str, Any]]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Evaluate one user-approved graph and yield SSE-friendly events."""
    del fedot_url, initial_fedot_config
    iterations = 1

    if task_type not in SUPPORTED_TASKS:
        yield _event("error", message=f"Unknown task type: {task_type}. Available: {SUPPORTED_TASKS}")
        return
    if task_type == "ts_forecasting" and not forecast_length:
        yield _event("error", message="ts_forecasting requires forecast_length")
        return
    csv_path = normalize_csv_path(csv_path)

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
        primary_metric=primary_metric,
        test_size=test_size,
        industrial_strategy=(str(industrial_strategy or "default").strip().lower() or "default"),
        industrial_strategy_params=dict(industrial_strategy_params or {}),
    )
    normalized_previous_evaluations = _normalize_previous_evaluations(previous_evaluations or [])
    data_context.iteration_history = _iteration_records_from_previous_evaluations(normalized_previous_evaluations)
    yield _event(
        "status",
        message=f"Data profiled: {profile.get('n_samples')} samples, {profile.get('n_features')} numeric features",
    )
    yield _event("status", message=f"Train/test split: test_size={test_size:.2f}")
    if primary_metric:
        yield _event("status", message=f"Primary metric: {primary_metric}")
    if data_context.industrial_strategy != "default":
        yield _event(
            "status",
            message=(
                f"Industrial execution strategy: {data_context.industrial_strategy}. "
                "Fedot.Industrial will run the strategy-specific training path."
            ),
        )
    else:
        yield _event(
            "status",
            message=(
                "Industrial execution strategy: default (Fedot.Industrial default strategy; "
                "task data_type is passed through Fedot.Industrial config)."
            ),
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
                reasoning="This evaluation trains the approved graph without structural changes.",
            )

        for iteration in range(1, max(1, iterations) + 1):
            yield _event("status", message="Evaluating the approved pipeline")

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
                if not architect_result.graph:
                    yield _event(
                        "error",
                        message="Architect LLM did not produce a valid graph proposal.",
                        diagnostics=architect_result.diagnostics,
                    )
                    return
                yield _event(
                    "agent_done",
                    agent="Architect",
                    iteration=iteration,
                    summary=f"Graph nodes: {len(architect_result.graph.get('nodes', []))}",
                    graph=architect_result.graph,
                    mermaid=architect_result.mermaid,
                    diagnostics=architect_result.diagnostics,
                    tool_calls_count=len(architect_result.tool_calls),
                )

                yield _event("agent_start", agent="Engineer", iteration=iteration, step="2/3")
                strategy_name = data_context.industrial_strategy or "default"
                logger.info(
                    "Engineer training started: iteration=%s strategy=%s task=%s metric=%s",
                    iteration,
                    strategy_name,
                    data_context.task_type,
                    data_context.primary_metric or "",
                )
                industrial_fit_only = _graph_has_industrial_native_model(architect_result.graph)
                if industrial_fit_only:
                    yield _event(
                        "status",
                        message=(
                            "Engineer is fitting the Industrial-native model through Fedot.Industrial. "
                            "The tuner is skipped for this graph."
                        ),
                    )
                elif strategy_name != "default":
                    yield _event(
                        "status",
                        message=(
                            f"Engineer is running Fedot.Industrial strategy '{strategy_name}'. "
                            "Finetune of the assumption graph can take longer than direct graph execution."
                        ),
                    )
                else:
                    yield _event(
                        "status",
                        message=(
                            "Engineer is finetuning the proposed graph through Fedot.Industrial AutoML "
                            "(default execution mode)."
                        ),
                    )
                engineer_task = asyncio.create_task(
                    asyncio.to_thread(_execute_engineer_sync, engineer, architect_result, data_context)
                )
                engineer_started = time.monotonic()
                while not engineer_task.done():
                    await asyncio.sleep(15)
                    if engineer_task.done():
                        break
                    elapsed = int(time.monotonic() - engineer_started)
                    if industrial_fit_only:
                        message = "Still fitting the Industrial-native model through Fedot.Industrial."
                    elif strategy_name != "default":
                        message = (
                            f"Still running '{strategy_name}'. Fedot.Industrial may be fitting "
                            "several internal branch pipelines."
                        )
                    else:
                        message = "Still finetuning the assumption graph through Fedot.Industrial AutoML."
                    logger.info(
                        "Engineer training progress: iteration=%s elapsed=%ss message=%s",
                        iteration,
                        elapsed,
                        message,
                    )
                    yield _event(
                        "agent_progress",
                        agent="Engineer",
                        iteration=iteration,
                        elapsed_seconds=elapsed,
                        message=message,
                    )
                engineer_result = await engineer_task
                logger.info(
                    "Engineer training completed: iteration=%s score=%.4f strategy=%s",
                    iteration,
                    engineer_result.graph_score,
                    engineer_result.industrial_strategy or strategy_name,
                )
                yield _event(
                    "agent_done",
                    agent="Engineer",
                    iteration=iteration,
                    summary=_engineer_summary(engineer_result),
                    diagnostics=engineer_result.diagnostics,
                    graph_error=engineer_result.graph_error,
                    tool_calls_count=len(engineer_result.tool_calls),
                )
                if engineer_result.diagnostics:
                    yield _event(
                        "diagnostics",
                        agent="Engineer",
                        iteration=iteration,
                        diagnostics=engineer_result.diagnostics,
                    )

                yield _event("agent_start", agent="Critic", iteration=iteration, step="3/3")
                critic_result = await critic.execute(architect_result, engineer_result, data_context, iteration)
                yield _event(
                    "agent_done",
                    agent="Critic",
                    iteration=iteration,
                    summary=(
                        f"Decision: {critic_result.winner}; "
                        f"suggested changes={len(critic_result.suggested_mutations)}"
                    ),
                    diagnostics=critic_result.diagnostics,
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
                        winner=critic_result.winner,
                        suggested_mutations=critic_result.suggested_mutations,
                    )
                )

                yield _event(
                    "iteration_done",
                    iteration=iteration,
                    graph_score=engineer_result.graph_score,
                    primary_metric=engineer_result.graph_metrics.get("primary_metric"),
                    primary_metric_value=engineer_result.graph_metrics.get("primary_metric_value"),
                    winner=critic_result.winner,
                    graph=architect_result.graph,
                    mermaid=architect_result.mermaid,
                    assumption_graph=engineer_result.assumption_graph,
                    assumption_mermaid=engineer_result.assumption_mermaid,
                    industrial_strategy=engineer_result.industrial_strategy,
                    industrial_strategy_params=engineer_result.industrial_strategy_params,
                    diagnostics=engineer_result.diagnostics + critic_result.diagnostics,
                )

                prev_feedback = critic_result
                prev_graph = architect_result.graph

            except Exception as exc:
                logger.exception("Evaluation %s failed", iteration)
                all_results.append({"iteration": iteration, "error": str(exc), "status": "failed"})
                yield _event("error", message=f"Evaluation failed: {str(exc)[:200]}")

        yield _event("agent_start", agent="Scribe", iteration=0, step="final")
        previous_evaluations = normalized_previous_evaluations
        best_evaluation = _get_best_evaluation(all_results, previous_evaluations)
        current_vs_best = _compare_current_to_best(all_results, best_evaluation)
        scribe_result = await scribe.execute(
            all_results,
            data_context,
            previous_evaluations=previous_evaluations,
            best_evaluation=best_evaluation,
            current_vs_best=current_vs_best,
        )
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
            "primary_metric": primary_metric,
            "is_time_series": is_ts_task(task_type),
            "forecast_length": forecast_length,
            "profile": profile,
            "iterations": all_results,
            "best_iteration": _get_best_iteration(all_results),
            "previous_evaluations": previous_evaluations,
            "best_evaluation": best_evaluation,
            "current_vs_best": current_vs_best,
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
    primary_metric: Optional[str] = None,
    test_size: float = 0.2,
    industrial_strategy: str = "default",
    industrial_strategy_params: Optional[Dict[str, Any]] = None,
    previous_evaluations: Optional[List[Dict[str, Any]]] = None,
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
        primary_metric=primary_metric,
        test_size=test_size,
        industrial_strategy=industrial_strategy,
        industrial_strategy_params=industrial_strategy_params,
        previous_evaluations=previous_evaluations,
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
    primary_metric: Optional[str] = None,
    industrial_strategy: str = "default",
    industrial_strategy_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One-shot Architect interaction for the frontend graph approval flow."""
    csv_path = normalize_csv_path(csv_path)
    if task_type not in SUPPORTED_TASKS:
        return {"error": f"Unknown task type: {task_type}"}

    profile = _profile_data(csv_path, target_column, task_type, forecast_length)
    data_context = DataContext(
        csv_path=csv_path,
        target_column=target_column,
        task_type=task_type,
        profile=profile,
        forecast_length=forecast_length,
        primary_metric=primary_metric,
        industrial_strategy=(str(industrial_strategy or "default").strip().lower() or "default"),
        industrial_strategy_params=dict(industrial_strategy_params or {}),
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
        if not result.graph:
            return {
                "error": "Architect LLM did not produce a valid graph proposal.",
                "profile": profile,
                "diagnostics": result.diagnostics,
                "tool_calls": [tc.to_dict() for tc in result.tool_calls],
            }
        return {
            "profile": profile,
            "graph": result.graph,
            "mermaid": result.mermaid,
            "analysis": result.analysis,
            "reasoning": result.reasoning,
            "diagnostics": result.diagnostics,
            "tool_calls": [tc.to_dict() for tc in result.tool_calls],
        }
    finally:
        await mcp_client.cleanup()


async def propose_revision_from_critic(
    csv_path: str,
    target_column: str,
    task_type: str,
    current_graph: Dict[str, Any],
    critic_feedback: Dict[str, Any],
    message: str = "",
    forecast_length: Optional[int] = None,
    primary_metric: Optional[str] = None,
    selected_mutations: Optional[list] = None,
    industrial_strategy: str = "default",
    industrial_strategy_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a new draft graph from explicit Critic feedback; user must approve it.

    ``selected_mutations`` lets the frontend pass the subset the user picked via
    checkboxes. Architect receives the graph-level mutations as LLM context and
    returns the revised graph. Strategy changes are selected outside Critic and
    outside the graph."""
    csv_path = normalize_csv_path(csv_path)
    if task_type not in SUPPORTED_TASKS:
        return {"error": f"Unknown task type: {task_type}"}

    profile = _profile_data(csv_path, target_column, task_type, forecast_length)
    graph = PipelineGraph.from_dict(current_graph)
    ok, validation_message = graph.validate()
    if not ok:
        return {"error": f"Current graph is invalid: {validation_message}"}

    selected_feedback_mutations = (
        [m for m in selected_mutations if isinstance(m, dict)]
        if selected_mutations is not None
        else critic_feedback.get("suggested_mutations", []) or []
    )

    graph_mutations, applied_strategy, applied_strategy_params = _split_graph_and_strategy_mutations(
        selected_feedback_mutations,
        fallback_strategy=industrial_strategy,
        fallback_params=industrial_strategy_params,
    )

    if graph_mutations:
        try:
            draft_graph = graph
            for mutation in graph_mutations:
                draft_graph = draft_graph.apply_mutation(mutation)
                ok, validation_message = draft_graph.validate()
                if not ok:
                    return {
                        "error": f"Selected Critic mutation produced an invalid graph: {validation_message}",
                        "profile": profile,
                        "diagnostics": [
                            {
                                "agent": "Architect",
                                "kind": "invalid_selected_mutation",
                                "summary": "Selected Critic graph mutation could not be applied safely.",
                                "technical_message": validation_message,
                                "recoverable": True,
                            }
                        ],
                        "tool_calls": [],
                    }
            return {
                "profile": profile,
                "graph": draft_graph.to_dict(),
                "mermaid": draft_graph.to_mermaid(),
                "analysis": "Applied the selected Critic graph mutation(s) directly as a new draft.",
                "reasoning": (
                    "Critic mutations are structural node edits, so the backend applied them deterministically "
                    "instead of asking Architect to reinterpret the same change."
                ),
                "diagnostics": [],
                "applied_mutations": selected_feedback_mutations,
                "industrial_strategy": applied_strategy or "default",
                "industrial_strategy_params": dict(applied_strategy_params or {}),
                "requires_approval": True,
                "tool_calls": [],
            }
        except Exception as exc:
            return {
                "error": f"Selected Critic mutation could not be applied: {exc}",
                "profile": profile,
                "diagnostics": [
                    {
                        "agent": "Architect",
                        "kind": "selected_mutation_error",
                        "summary": "Selected Critic graph mutation failed before a draft could be prepared.",
                        "technical_message": str(exc),
                        "recoverable": True,
                    }
                ],
                "tool_calls": [],
            }

    data_context = DataContext(
        csv_path=csv_path,
        target_column=target_column,
        task_type=task_type,
        profile=profile,
        forecast_length=forecast_length,
        primary_metric=primary_metric,
        industrial_strategy=(str(applied_strategy or "default").strip().lower() or "default"),
        industrial_strategy_params=dict(applied_strategy_params or {}),
    )
    feedback = CriticFeedback(
        winner=critic_feedback.get("winner", "needs_revision"),
        assessment=critic_feedback.get("assessment", ""),
        strengths=critic_feedback.get("strengths", []) or [],
        weaknesses=critic_feedback.get("weaknesses", []) or [],
        suggested_mutations=graph_mutations,
        improvement_plan=critic_feedback.get("improvement_plan", []) or [],
        questions_for_user=critic_feedback.get("questions_for_user", []) or [],
        should_stop=False,
        diagnostics=critic_feedback.get("diagnostics", []) or [],
    )
    if message:
        feedback.weaknesses.append(f"User note: {message}")

    mcp_client = await _connect_mcp()
    try:
        architect = Architect(mcp_client=mcp_client)
        result = await architect.execute(
            data_context=data_context,
            iteration=1,
            prev_feedback=feedback,
            prev_graph=current_graph,
        )
        if not result.graph:
            return {
                "error": "Architect LLM did not produce a valid revised graph proposal.",
                "profile": profile,
                "diagnostics": result.diagnostics,
                "tool_calls": [tc.to_dict() for tc in result.tool_calls],
            }
        return {
            "profile": profile,
            "graph": result.graph,
            "mermaid": result.mermaid or PipelineGraph.from_dict(result.graph).to_mermaid(),
            "analysis": result.analysis,
            "reasoning": result.reasoning,
            "diagnostics": result.diagnostics,
            "applied_mutations": selected_feedback_mutations,
            "industrial_strategy": data_context.industrial_strategy,
            "industrial_strategy_params": dict(data_context.industrial_strategy_params or {}),
            "requires_approval": True,
            "tool_calls": [tc.to_dict() for tc in result.tool_calls],
        }
    except Exception:
        logger.exception("Architect revision proposal failed")
    finally:
        await mcp_client.cleanup()

    return {
        "error": "Architect revision proposal failed.",
        "profile": profile,
        "diagnostics": [],
        "tool_calls": [],
    }


def _split_graph_and_strategy_mutations(
    mutations: list,
    fallback_strategy: str = "default",
    fallback_params: Optional[Dict[str, Any]] = None,
):
    """Filter Critic suggestions down to graph node mutations.

    Critic is graph-structural only. Strategy switches and node params are
    ignored here; strategy selection lives in the dedicated UI controls.
    Returns (graph_mutations, strategy_name, strategy_params)."""
    graph_mutations = []
    chosen_strategy = str(fallback_strategy or "default").strip().lower() or "default"
    chosen_params: Dict[str, Any] = dict(fallback_params or {})
    for mutation in mutations or []:
        if not isinstance(mutation, dict):
            continue
        if mutation.get("type") == "set_strategy":
            continue
        cleaned = dict(mutation)
        cleaned.pop("params", None)
        node = cleaned.get("node")
        if isinstance(node, dict):
            cleaned["node"] = {k: v for k, v in node.items() if k != "params"}
        graph_mutations.append(cleaned)
    return graph_mutations, chosen_strategy, chosen_params


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


def _metric_value_from_evaluation(item: Dict[str, Any]) -> Optional[float]:
    if not isinstance(item, dict):
        return None
    if item.get("score") is not None:
        try:
            return float(item.get("score"))
        except (TypeError, ValueError):
            pass
    engineer = item.get("engineer", {}) or {}
    metrics = engineer.get("test_metrics", {}) or engineer.get("graph_metrics", {}) or {}
    value = metrics.get("primary_metric_value", engineer.get("graph_score"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_name_from_evaluation(item: Dict[str, Any], fallback: str = "score") -> str:
    if not isinstance(item, dict):
        return fallback
    if item.get("primary_metric"):
        return str(item.get("primary_metric"))
    engineer = item.get("engineer", {}) or {}
    metrics = engineer.get("test_metrics", {}) or engineer.get("graph_metrics", {}) or {}
    return str(metrics.get("primary_metric") or fallback)


def _graph_from_evaluation(item: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    if isinstance(item.get("graph"), dict):
        return item["graph"]
    architect = item.get("architect", {}) or {}
    engineer = item.get("engineer", {}) or {}
    return architect.get("graph") or engineer.get("assumption_graph") or {}


def _compact_current_evaluation(item: Dict[str, Any], source: str = "current") -> Dict[str, Any]:
    engineer = item.get("engineer", {}) or {}
    metrics = engineer.get("test_metrics", {}) or engineer.get("graph_metrics", {}) or {}
    train_metrics = engineer.get("train_metrics", {}) or {}
    critic = item.get("critic", {}) or {}
    return {
        "source": source,
        "iteration": item.get("iteration"),
        "label": item.get("label") or source,
        "score": _metric_value_from_evaluation(item),
        "primary_metric": _metric_name_from_evaluation(item, "score"),
        "test_metrics": metrics,
        "train_metrics": train_metrics,
        "critic_decision": critic.get("winner", ""),
        "suggested_mutations": critic.get("suggested_mutations", []) or [],
        "graph": _graph_from_evaluation(item),
        "summary": item.get("summary", ""),
    }


def _normalize_previous_evaluations(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for idx, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        score = _metric_value_from_evaluation(item)
        if score is None:
            continue
        normalized.append({
            "source": item.get("source") or item.get("reason") or f"previous_{idx + 1}",
            "saved_at": item.get("saved_at", ""),
            "label": item.get("label", ""),
            "score": score,
            "primary_metric": _metric_name_from_evaluation(item, "score"),
            "test_metrics": item.get("test_metrics", {}) or {},
            "train_metrics": item.get("train_metrics", {}) or {},
            "critic_decision": item.get("critic_decision", ""),
            "suggested_mutations": item.get("suggested_mutations", []) or [],
            "graph": _graph_from_evaluation(item),
            "summary": item.get("summary", ""),
        })
    return normalized[:10]


def _iteration_records_from_previous_evaluations(items: List[Dict[str, Any]]) -> List[IterationRecord]:
    records: List[IterationRecord] = []
    for idx, item in enumerate(items or []):
        graph = _graph_from_evaluation(item)
        if not graph:
            continue
        records.append(
            IterationRecord(
                iteration=-(idx + 1),
                graph=graph,
                graph_score=float(item.get("score") or 0.0),
                winner=str(item.get("critic_decision") or ""),
                suggested_mutations=[
                    mutation
                    for mutation in (item.get("suggested_mutations", []) or [])
                    if isinstance(mutation, dict)
                ],
            )
        )
    return records[:10]


def _get_best_evaluation(current_results: List[Dict[str, Any]], previous_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = list(previous_evaluations or [])
    candidates.extend(
        _compact_current_evaluation(item, source="current_run")
        for item in current_results
        if isinstance(item, dict) and "error" not in item
    )
    best, best_score = {}, None
    for item in candidates:
        score = _metric_value_from_evaluation(item)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best = item
    return best


def _compare_current_to_best(current_results: List[Dict[str, Any]], best_evaluation: Dict[str, Any]) -> Dict[str, Any]:
    current = _compact_current_evaluation(_get_best_iteration(current_results), source="current_run")
    current_score = _metric_value_from_evaluation(current)
    best_score = _metric_value_from_evaluation(best_evaluation)
    if current_score is None or best_score is None:
        return {"status": "unavailable"}
    delta = current_score - best_score
    return {
        "status": "current_is_best" if delta >= 0 else "current_below_best",
        "primary_metric": current.get("primary_metric") or best_evaluation.get("primary_metric") or "score",
        "current_score": current_score,
        "best_score": best_score,
        "delta": delta,
        "current_label": current.get("label", "current"),
        "best_label": best_evaluation.get("label") or best_evaluation.get("source") or "best",
        "best_source": best_evaluation.get("source", ""),
        "best_saved_at": best_evaluation.get("saved_at", ""),
    }


def _engineer_summary(engineer_result: EngineerResult) -> str:
    metric = engineer_result.graph_metrics.get("primary_metric") or "score"
    value = engineer_result.graph_metrics.get("primary_metric_value", engineer_result.graph_score)
    try:
        value_text = f"{float(value):.4f}"
    except (TypeError, ValueError):
        value_text = str(value)
    strategy = engineer_result.industrial_strategy or "default"
    prefix = f"{strategy}; " if strategy != "default" else ""
    return f"{prefix}{metric}: {value_text}; ranking score: {engineer_result.graph_score:.4f}"


def _create_summary(all_results, task_type):
    ok = [item for item in all_results if "error" not in item]
    if not ok:
        return {"status": "no successful evaluations", "task_type": task_type}

    graph_scores = [float(item.get("engineer", {}).get("graph_score", 0) or 0) for item in ok]
    return {
        "total_evaluations": len(all_results),
        "successful_evaluations": len(ok),
        "task_type": task_type,
        "is_time_series": is_ts_task(task_type),
        "avg_graph_score": sum(graph_scores) / len(graph_scores),
        "max_graph_score": max(graph_scores),
        "accepted_count": sum(
            1 for item in ok if item.get("critic", {}).get("winner") == "accepted"
        ),
    }
