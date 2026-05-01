import logging
import os
from typing import Any, AsyncGenerator, Dict, Optional

import pandas as pd

from agents import Architect, ArchitectResult, Critic, CriticFeedback, DataContext, Engineer, EngineerResult, IterationRecord, Scribe
from data_profiler import DataProfiler
from graph_engine import SUPPORTED_TASKS, PipelineGraph, diagnose_runtime_error, is_ts_task
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
    )
    yield _event(
        "status",
        message=f"Data profiled: {profile.get('n_samples')} samples, {profile.get('n_features')} numeric features",
    )
    if primary_metric:
        yield _event("status", message=f"Primary metric: {primary_metric}")

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
                engineer_result = await engineer.execute(architect_result, data_context)
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
                    diagnostics=engineer_result.diagnostics + critic_result.diagnostics,
                )

                prev_feedback = critic_result
                prev_graph = architect_result.graph

            except Exception as exc:
                logger.exception("Evaluation %s failed", iteration)
                all_results.append({"iteration": iteration, "error": str(exc), "status": "failed"})
                yield _event("error", message=f"Evaluation failed: {str(exc)[:200]}")

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
            "primary_metric": primary_metric,
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
    primary_metric: Optional[str] = None,
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
) -> Dict[str, Any]:
    """Create a new draft graph from explicit Critic feedback; user must approve it.

    Critic mutations are alternatives, not a sequence. ``selected_mutations`` lets
    the frontend pass the subset the user picked via checkboxes. If omitted, only
    the first suggested mutation is applied to avoid combining incompatible
    alternatives (e.g. xgboost-only params plus a replace to rf)."""
    csv_path = normalize_csv_path(csv_path)
    if task_type not in SUPPORTED_TASKS:
        return {"error": f"Unknown task type: {task_type}"}

    profile = _profile_data(csv_path, target_column, task_type, forecast_length)
    graph = PipelineGraph.from_dict(current_graph)
    ok, validation_message = graph.validate()
    if not ok:
        return {"error": f"Current graph is invalid: {validation_message}"}

    all_mutations = critic_feedback.get("suggested_mutations", []) or []
    if selected_mutations is not None:
        mutations = [m for m in selected_mutations if isinstance(m, dict)]
    else:
        mutations = all_mutations[:1]
    diagnostics = []
    draft = graph
    applied = []
    for mutation in mutations:
        try:
            candidate = draft.apply_mutation(mutation)
            ok, validation_message = candidate.validate()
            if ok:
                draft = candidate
                applied.append(mutation)
            else:
                diagnostics.append(diagnose_runtime_error(validation_message, task_type=task_type, graph=candidate))
        except Exception as exc:
            diagnostics.append(diagnose_runtime_error(exc, task_type=task_type, graph=draft))

    if applied:
        return {
            "profile": profile,
            "graph": draft.to_dict(),
            "mermaid": draft.to_mermaid(),
            "analysis": "Architect drafted a new graph by applying Critic feedback. It is not approved yet.",
            "reasoning": _revision_reasoning(critic_feedback, applied, message),
            "diagnostics": diagnostics,
            "applied_mutations": applied,
            "requires_approval": True,
            "tool_calls": [],
        }

    data_context = DataContext(
        csv_path=csv_path,
        target_column=target_column,
        task_type=task_type,
        profile=profile,
        forecast_length=forecast_length,
        primary_metric=primary_metric,
    )
    feedback = CriticFeedback(
        winner=critic_feedback.get("winner", "needs_revision"),
        assessment=critic_feedback.get("assessment", ""),
        strengths=critic_feedback.get("strengths", []) or [],
        weaknesses=critic_feedback.get("weaknesses", []) or [],
        suggested_mutations=[],
        improvement_plan=critic_feedback.get("improvement_plan", []) or [],
        should_stop=False,
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
        revised_graph = result.graph
        heuristic_applied = []
        if _same_graph(revised_graph, current_graph):
            heuristic_mutation = _heuristic_revision_mutation(graph, profile, critic_feedback)
            if heuristic_mutation:
                candidate = graph.apply_mutation(heuristic_mutation)
                ok, validation_message = candidate.validate()
                if ok:
                    revised_graph = candidate.to_dict()
                    heuristic_applied.append(heuristic_mutation)
                else:
                    diagnostics.append(diagnose_runtime_error(validation_message, task_type=task_type, graph=candidate))
        return {
            "profile": profile,
            "graph": revised_graph,
            "mermaid": PipelineGraph.from_dict(revised_graph).to_mermaid(),
            "analysis": "Architect drafted a revised graph from Critic feedback. It is not approved yet.",
            "reasoning": result.reasoning or _revision_reasoning(critic_feedback, heuristic_applied, message),
            "diagnostics": diagnostics + result.diagnostics,
            "applied_mutations": heuristic_applied,
            "requires_approval": True,
            "tool_calls": [tc.to_dict() for tc in result.tool_calls],
        }
    except Exception as exc:
        logger.exception("Architect revision proposal failed")
        diagnostics.append(diagnose_runtime_error(exc, task_type=task_type, graph=graph))
    finally:
        await mcp_client.cleanup()

    return {
        "profile": profile,
        "graph": graph.to_dict(),
        "mermaid": graph.to_mermaid(),
        "analysis": "Critic did not provide a valid structural mutation, so Architect kept the current graph as the draft.",
        "reasoning": _revision_reasoning(critic_feedback, [], message),
        "diagnostics": diagnostics,
        "applied_mutations": [],
        "requires_approval": True,
        "tool_calls": [],
    }


def _revision_reasoning(critic_feedback: Dict[str, Any], applied: list, message: str = "") -> str:
    parts = []
    assessment = critic_feedback.get("assessment")
    if assessment:
        parts.append(f"Critic assessment: {assessment}")
    plan = critic_feedback.get("improvement_plan", []) or []
    if plan:
        parts.append("Improvement plan: " + " ".join(str(item) for item in plan[:4]))
    parts.append(f"Applied mutations: {applied}")
    if message:
        parts.append(f"User note: {message}")
    return "\n".join(parts)


def _same_graph(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return json.dumps(left or {}, sort_keys=True) == json.dumps(right or {}, sort_keys=True)


def _heuristic_revision_mutation(
    graph: PipelineGraph,
    profile: Dict[str, Any],
    critic_feedback: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if graph.task_type != "classification":
        return None
    root_id = graph.root_id()
    root = next((node for node in graph.nodes if node.id == root_id), None)
    if root is None:
        return None
    feedback_text = " ".join(
        str(item)
        for item in [
            critic_feedback.get("assessment", ""),
            *(critic_feedback.get("weaknesses", []) or []),
            *(critic_feedback.get("improvement_plan", []) or []),
        ]
    ).lower()
    wants_balance = profile.get("is_imbalanced") or "imbalance" in feedback_text or "class weight" in feedback_text
    wants_regularization = "variance" in feedback_text or "regularization" in feedback_text

    params: Dict[str, Any] = {}
    if wants_balance:
        try:
            ratio = float(profile.get("imbalance_ratio") or 0)
        except (TypeError, ValueError):
            ratio = 0
        weight = round(min(max(1.0 / ratio, 1.0), 20.0), 3) if ratio else 3.0
        if root.operation == "xgboost":
            params["scale_pos_weight"] = weight
        elif root.operation in ("rf", "logit", "lgbm", "dt"):
            params["class_weight"] = "balanced"
            if root.operation == "logit":
                params["max_iter"] = 1000

    if wants_regularization:
        params.update({
            "xgboost": {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200},
            "rf": {"max_depth": 8, "n_estimators": 300},
            "logit": {"C": 0.5, "max_iter": 1000},
            "lgbm": {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 200},
            "dt": {"max_depth": 6},
        }.get(root.operation, {}))

    if params:
        return {"type": "set_params", "node_id": root_id, "params": params}
    return None


def mutate_graph_locally(graph: Dict[str, Any], mutation: Dict[str, Any]) -> Dict[str, Any]:
    pipeline_graph = PipelineGraph.from_dict(graph)
    mutated = pipeline_graph.apply_mutation(mutation)
    ok, message = mutated.validate()
    diagnostics = [] if ok else [diagnose_runtime_error(message, task_type=mutated.task_type, graph=mutated)]
    return {
        "valid": ok,
        "message": message,
        "graph": mutated.to_dict(),
        "mermaid": mutated.to_mermaid() if ok else "",
        "diagnostics": diagnostics,
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


def _engineer_summary(engineer_result: EngineerResult) -> str:
    metric = engineer_result.graph_metrics.get("primary_metric") or "score"
    value = engineer_result.graph_metrics.get("primary_metric_value", engineer_result.graph_score)
    try:
        value_text = f"{float(value):.4f}"
    except (TypeError, ValueError):
        value_text = str(value)
    return f"{metric}: {value_text}; ranking score: {engineer_result.graph_score:.4f}"


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
