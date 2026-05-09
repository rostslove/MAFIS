import json
import os
import copy
import hashlib
import html
import io
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st


APP_NAME = "MultiAgentFedot.IndustrialSystem"
APP_SHORT_NAME = "MAFIS"

st.set_page_config(page_title=APP_SHORT_NAME, layout="wide")

st.markdown(
    """
    <style>
    .run-log, .training-journal {
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
        margin: 0.35rem 0 0.75rem 0;
    }
    .run-log-entry, .training-journal-entry {
        border: 1px solid #e3e7ef;
        border-left: 3px solid #5876d6;
        border-radius: 8px;
        background: #ffffff;
        padding: 0.55rem 0.7rem;
    }
    .run-log-entry.done, .training-journal-entry.done {
        border-left-color: #228b5b;
        background: #f7fbf8;
    }
    .run-log-entry.warn, .training-journal-entry.warn {
        border-left-color: #c27a1a;
        background: #fffaf1;
    }
    .run-log-entry.error, .training-journal-entry.error {
        border-left-color: #c24141;
        background: #fff7f7;
    }
    .run-log-label, .training-journal-label {
        color: #3a4253;
        font-size: 0.74rem;
        font-weight: 700;
        line-height: 1.15;
        text-transform: uppercase;
    }
    .run-log-text, .training-journal-text {
        color: #1f2937;
        font-size: 0.9rem;
        line-height: 1.35;
        margin-top: 0.15rem;
    }
    .training-journal-meta {
        color: #5f6b7a;
        font-size: 0.78rem;
        line-height: 1.25;
        margin-top: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")
TASK_TYPES = ["classification", "regression", "ts_classification", "ts_regression", "ts_forecasting"]
DEFAULT_METRICS_BY_TASK = {
    "classification": ["roc_auc", "f1", "accuracy", "precision"],
    "regression": ["r2", "rmse", "mse", "mae"],
    "ts_classification": ["f1", "accuracy", "roc_auc"],
    "ts_regression": ["r2", "rmse", "mae"],
    "ts_forecasting": ["rmse", "mae", "mape", "smape"],
}
DEFAULT_M4_GROUPS = ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
METRIC_METADATA_KEYS = {
    "error",
    "primary_metric",
    "primary_metric_value",
    "primary_score",
    "primary_score_direction",
}


def shared_data_dir() -> Path:
    docker_dir = Path("/app/data")
    if docker_dir.exists():
        return docker_dir
    repo_data = Path(__file__).resolve().parents[1] / "data"
    repo_data.mkdir(exist_ok=True)
    return repo_data


def load_config() -> Dict[str, Any]:
    try:
        response = requests.get(f"{BACKEND_URL}/config", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {"operations": {}, "metrics_by_task": DEFAULT_METRICS_BY_TASK, "supported_tasks": TASK_TYPES}


def load_tools() -> Dict[str, Any]:
    try:
        response = requests.get(f"{BACKEND_URL}/tools", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


def post_json(path: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    response = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            body = response.json()
            detail = body.get("detail") or body.get("error") or response.text
            diagnostics = body.get("diagnostics") or []
            if diagnostics:
                summaries = "; ".join(item.get("summary", "") for item in diagnostics[:3] if item.get("summary"))
                if summaries:
                    detail = f"{detail} Diagnostics: {summaries}"
        except Exception:
            detail = response.text
        raise RuntimeError(f"{response.status_code}: {detail}") from exc
    return response.json()


def graph_to_dot(graph: Dict[str, Any]) -> str:
    nodes = graph.get("nodes", [])
    lines = ["digraph G {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fillcolor="#f7f7f7"];']
    for node in nodes:
        node_id = node.get("id", "")
        label = f"{node_id}\\n{node.get('operation', '')}"
        lines.append(f'"{node_id}" [label="{label}"];')
    for node in nodes:
        for source in node.get("inputs", []):
            lines.append(f'"{source}" -> "{node.get("id", "")}";')
    lines.append("}")
    return "\n".join(lines)


def save_uploaded_csv(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    signature = f"{uploaded_file.name}:{hashlib.sha256(data).hexdigest()}"
    if (
        st.session_state.get("uploaded_signature") == signature
        and st.session_state.get("csv_path")
        and "df" in st.session_state
    ):
        return st.session_state.csv_path

    df = pd.read_csv(io.BytesIO(data))
    st.session_state.df = df
    st.session_state.result = {}
    st.session_state.evaluation_history = []
    st.session_state.uploaded_signature = signature
    data_dir = shared_data_dir()
    filename = f"uploaded_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = data_dir / filename
    df.to_csv(path, index=False)
    st.session_state.csv_path = f"/app/data/{filename}" if Path("/app/data").exists() else str(path)
    return str(path)


def current_payload(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = {
        "csv_path": st.session_state.get("csv_path", ""),
        "target_column": st.session_state.get("target_column", ""),
        "task_type": st.session_state.get("task_type", "classification"),
        "primary_metric": st.session_state.get("primary_metric", ""),
        "test_size": float(st.session_state.get("test_size", 0.2)),
        "industrial_strategy": st.session_state.get("industrial_strategy") or "default",
        "industrial_strategy_params": dict(st.session_state.get("industrial_strategy_params") or {}),
    }
    if st.session_state.get("forecast_length"):
        payload["forecast_length"] = int(st.session_state.forecast_length)
    if extra:
        payload.update(extra)
    return payload


def set_graph(result: Dict[str, Any], approved: bool = False) -> None:
    st.session_state.graph = result.get("graph", {})
    st.session_state.mermaid = result.get("mermaid", "")
    if "analysis" in result:
        st.session_state.architect_analysis = result.get("analysis", "")
    if "reasoning" in result:
        st.session_state.architect_reasoning = result.get("reasoning", "")
    if "diagnostics" in result:
        st.session_state.architect_diagnostics = result.get("diagnostics", [])
    st.session_state.graph_approved = approved
    if approved:
        st.session_state.approved_graph = st.session_state.graph
        st.session_state.approved_mermaid = st.session_state.mermaid


def approve_graph() -> None:
    if st.session_state.get("graph"):
        st.session_state.approved_graph = st.session_state.graph
        st.session_state.approved_mermaid = st.session_state.get("mermaid", "")
        st.session_state.graph_approved = True
        st.session_state.revision_success_message = ""


def discard_draft() -> None:
    approved = st.session_state.get("approved_graph")
    if approved:
        st.session_state.graph = approved
        st.session_state.mermaid = st.session_state.get("approved_mermaid", "")
        st.session_state.graph_approved = True
    st.session_state.revision_success_message = ""


def root_node_id(graph: Dict[str, Any]) -> str:
    nodes = graph.get("nodes", [])
    ids = {node.get("id") for node in nodes}
    children = {inp for node in nodes for inp in node.get("inputs", [])}
    roots = [node_id for node_id in ids - children if node_id]
    return roots[0] if roots else ""


def root_node_ids(graph: Dict[str, Any]) -> List[str]:
    nodes = graph.get("nodes", [])
    ids = {node.get("id") for node in nodes if node.get("id")}
    children = {inp for node in nodes for inp in node.get("inputs", [])}
    return [node_id for node_id in ids - children if node_id]


def adopt_graph_draft(graph: Dict[str, Any]) -> None:
    set_graph({"graph": graph, "mermaid": ""}, approved=False)


def require_dataset() -> bool:
    if not st.session_state.get("csv_path") or "df" not in st.session_state:
        st.info("Upload a CSV file to start, or use the Benchmarks tab to run M4 without uploading a dataset.")
        return False
    return True


def render_graph(graph: Dict[str, Any], show_details: bool = True) -> None:
    del show_details
    if not graph:
        st.info("No graph proposed yet.")
        return
    st.graphviz_chart(graph_to_dot(graph), use_container_width=True)


def render_tool_calls(tool_calls: List[Dict[str, Any]], title: str) -> None:
    if not tool_calls:
        return
    with st.expander(title):
        rows = [
            {
                "tool": call.get("tool"),
                "success": call.get("success"),
                "args": json.dumps(call.get("args", {}), ensure_ascii=False)[:160],
                "result": str(call.get("result", ""))[:200],
            }
            for call in tool_calls
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_diagnostics(diagnostics: List[Dict[str, Any]], title: str = "Diagnostics", use_expander: bool = True) -> None:
    if not diagnostics:
        return
    container = st.expander(title, expanded=True) if use_expander else st.container()
    with container:
        for item in diagnostics:
            st.warning(item.get("summary", "Runtime diagnostic"))
            recommendations = item.get("recommendations", []) or []
            if recommendations:
                st.write("Suggested fix")
                for rec in recommendations:
                    st.write(f"- {rec}")
            attempts = item.get("recovery_attempts") or []
            if attempts:
                with st.expander(f"Training attempts ({len(attempts)})", expanded=False):
                    st.dataframe(
                        pd.DataFrame(compact_training_attempt_rows(attempts)),
                        use_container_width=True,
                        hide_index=True,
                    )
            if item.get("runtime_issue"):
                st.caption(f"Runtime issue: `{item.get('runtime_issue')}`")
            if item.get("technical_message"):
                st.caption("Technical message")
                st.code(str(item["technical_message"])[:4000])


def compact_training_attempt_rows(attempts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    phase_labels = {
        "finetune_fit": "finetune fit",
        "finetune_predict": "finetune predict",
        "direct_fit": "direct-fit fallback",
        "train_graph": "train graph",
        "completed": "completed",
    }
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        error = (
            attempt.get("error")
            or attempt.get("technical_message")
            or attempt.get("finetune_error")
            or attempt.get("fallback_error")
            or ""
        )
        rows.append({
            "attempt": attempt.get("variant") or "attempt",
            "remaining graph": attempt.get("remaining_graph", ""),
            "stage": phase_labels.get(attempt.get("phase", ""), attempt.get("phase", "")),
            "score": attempt.get("score", ""),
            "fallback": attempt.get("fallback_used") or "",
            "error": str(error)[:300],
        })
    return rows


def unique_text(items: List[str]) -> List[str]:
    unique: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in unique:
            unique.append(text)
    return unique


def decision_label(value: Any) -> str:
    labels = {
        "needs_revision": "Need revision",
        "need_revision": "Need revision",
        "accepted": "Accepted",
    }
    text = str(value or "").strip()
    return labels.get(text, text)


def runtime_failure_details(engineer: Dict[str, Any], critic: Dict[str, Any]) -> Dict[str, Any]:
    error = str(engineer.get("graph_error") or "").strip()
    finetune_error = str(engineer.get("finetune_error") or "").strip()
    if not error and finetune_error:
        error = f"Fedot.Industrial finetune failed: {finetune_error}"
    if not error:
        return {}

    diagnostics = []
    diagnostics.extend(engineer.get("diagnostics", []) or [])
    diagnostics.extend(critic.get("diagnostics", []) or [])
    selected = next(
        (
            diagnostic
            for diagnostic in diagnostics
            if isinstance(diagnostic, dict) and diagnostic.get("kind") == "runtime_attempts_summary"
        ),
        {},
    )
    if not selected:
        selected = next(
            (
                diagnostic
                for diagnostic in diagnostics
                if isinstance(diagnostic, dict) and diagnostic.get("kind") == "failure_localization"
            ),
            {},
        )
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        kind = str(diagnostic.get("kind") or "")
        technical = str(diagnostic.get("technical_message") or "")
        if selected:
            break
        if technical == error or kind.endswith("_error") or kind in {
            "runtime_error", "finetune_fallback", "finetune_fallback_after_node_skip",
            "node_skip_recovery_failed", "node_skipped_after_runtime_error",
            "failure_localization",
        }:
            selected = diagnostic
            break
    if not selected and diagnostics:
        selected = next((d for d in diagnostics if isinstance(d, dict)), {})

    summary = str(selected.get("summary") or "").strip()
    technical = str(selected.get("technical_message") or error).strip()
    if finetune_error and finetune_error not in technical:
        technical = (
            "Fedot.Industrial finetune error:\n"
            f"{finetune_error}\n\n"
            "Engineer error:\n"
            f"{technical}"
        )
    if not summary or summary == technical:
        # Build a more helpful summary instead of a generic "training failed"
        if "finetune failed" in error.lower() or finetune_error:
            summary = (
                "Fedot.Industrial finetune raised an exception. "
                "Engineer attempted node-skip recovery; if recovery did not converge, "
                "the metrics below come from the direct-fit fallback path."
            )
        else:
            summary = "Training did not complete cleanly. See technical details below."
    attempt_diagnostic = next(
        (
            diagnostic
            for diagnostic in diagnostics
            if isinstance(diagnostic, dict) and diagnostic.get("kind") == "runtime_attempts_summary"
        ),
        {},
    )
    recommendations = unique_text(
        (selected.get("recommendations", []) or [])
        + (attempt_diagnostic.get("recommendations", []) or [])
    )
    recovery_attempts = selected.get("recovery_attempts") or attempt_diagnostic.get("recovery_attempts") or []
    fallback_used = str(engineer.get("fallback_used") or "").strip()
    return {
        "summary": summary,
        "technical": technical,
        "recommendations": recommendations,
        "recovery_attempts": recovery_attempts,
        "finetune_error": finetune_error,
        "fallback_used": fallback_used,
        "severity": "error" if engineer.get("graph_error") else "warning",
    }


def render_runtime_failure_once(iteration: Dict[str, Any]) -> bool:
    engineer = iteration.get("engineer", {}) or {}
    critic = iteration.get("critic", {}) or {}
    details = runtime_failure_details(engineer, critic)
    if not details:
        return False

    if details.get("severity") == "error":
        st.error(details["summary"])
    else:
        st.warning(details["summary"])
    fallback_used = details.get("fallback_used")
    if fallback_used:
        st.warning(
            f"Engineer used the **{fallback_used}** path as a baseline; metrics shown below come from that fallback, not from a successful Fedot.Industrial finetune."
        )
    recommendations = details.get("recommendations", []) or []
    if recommendations:
        st.write("Suggested fix")
        for recommendation in recommendations[:4]:
            st.write(f"- {recommendation}")
    recovery_attempts = details.get("recovery_attempts") or []
    if recovery_attempts:
        with st.expander(f"Training attempts ({len(recovery_attempts)})", expanded=False):
            st.dataframe(
                pd.DataFrame(compact_training_attempt_rows(recovery_attempts)),
                use_container_width=True,
                hide_index=True,
            )
    if details.get("technical"):
        # Open by default so the user immediately sees what actually broke.
        with st.expander("Technical exception (full message)", expanded=True):
            st.code(details["technical"][:6000])
    return True


def operation_rows(config: Dict[str, Any], task_type: str) -> List[Dict[str, str]]:
    catalog = config.get("operation_catalog", {}).get(task_type, {})
    if catalog:
        rows = []
        for group, items in catalog.items():
            for item in items:
                param_hints = item.get("param_hints", []) or []
                rows.append({
                    "group": group,
                    "operation": item.get("operation", ""),
                    "description": item.get("description", ""),
                    "source": item.get("source", ""),
                    "meta": item.get("fedot_industrial_meta", ""),
                    "tags": ", ".join(item.get("tags", []) or []),
                    "params": ", ".join(hint.get("name", "") for hint in param_hints) or "-",
                })
        return rows

    operations = config.get("operations", {}).get(task_type, {})
    descriptions = config.get("operation_descriptions", {})
    return [
        {"group": group, "operation": name, "description": descriptions.get(name, ""), "params": "-"}
        for group, names in operations.items()
        for name in names
    ]


def format_params(params: Dict[str, Any]) -> str:
    if not params:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in params.items())


def format_number(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def format_elapsed(seconds: Any) -> str:
    try:
        total = max(0, int(seconds))
    except (TypeError, ValueError):
        total = 0
    minutes, sec = divmod(total, 60)
    if minutes:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def _escape(value: Any) -> str:
    return html.escape(str(value or ""))


def _log_state(text: str) -> str:
    lower = text.lower()
    if "error" in lower or "failed" in lower or "exception" in lower:
        return "error"
    if "fallback" in lower or "revision" in lower or "warning" in lower:
        return "warn"
    if "done" in lower or "completed" in lower or "accepted" in lower:
        return "done"
    return "running"


def _split_log_line(line: str) -> tuple[str, str]:
    if ":" not in line:
        return "System", line
    label, text = line.split(":", 1)
    return label.strip() or "System", text.strip()


def render_run_log(lines: List[str]) -> None:
    if not lines:
        return
    blocks = []
    for line in lines[-14:]:
        label, text = _split_log_line(str(line))
        state = _log_state(line)
        blocks.append(
            "<div class='run-log-entry {state}'>"
            "<div class='run-log-label'>{label}</div>"
            "<div class='run-log-text'>{text}</div>"
            "</div>".format(
                state=_escape(state),
                label=_escape(label),
                text=_escape(text),
            )
        )
    st.markdown("<div class='run-log'>" + "".join(blocks) + "</div>", unsafe_allow_html=True)


def render_training_journal(engineer: Dict[str, Any]) -> None:
    log_rows = engineer.get("training_log", []) or []
    notes = unique_text(engineer.get("training_notes", []) or [])
    if not log_rows and not notes:
        return

    blocks = []
    split = engineer.get("split_info", {}) or {}
    strategy = engineer.get("industrial_strategy") or "default"
    context = engineer.get("industrial_repository_context") or "fedot_industrial"
    data_type = engineer.get("industrial_data_type") or ""
    n_jobs = engineer.get("n_jobs") or (engineer.get("industrial_strategy_params", {}) or {}).get("n_jobs")

    summary_bits = [f"strategy={strategy}", f"context={context}"]
    if data_type:
        summary_bits.append(f"data_type={data_type}")
    if n_jobs:
        summary_bits.append(f"n_jobs={n_jobs}")
    if split:
        summary_bits.append(f"train={split.get('n_train', '')}")
        summary_bits.append(f"holdout={split.get('n_test', '')}")
    blocks.append(
        "<div class='training-journal-entry done'>"
        "<div class='training-journal-label'>Execution</div>"
        "<div class='training-journal-text'>Fedot.Industrial training path</div>"
        "<div class='training-journal-meta'>{meta}</div>"
        "</div>".format(meta=_escape(", ".join(bit for bit in summary_bits if bit and not bit.endswith("="))))
    )

    for row in log_rows:
        if not isinstance(row, dict):
            continue
        stage = row.get("stage") or "stage"
        status = row.get("status") or "running"
        elapsed = row.get("elapsed_seconds")
        details = row.get("details") or {}
        meta_parts = []
        if elapsed is not None:
            meta_parts.append(f"elapsed={format_elapsed(elapsed)}")
        if isinstance(details, dict):
            meta_parts.extend(f"{key}={value}" for key, value in details.items())
        blocks.append(
            "<div class='training-journal-entry {state}'>"
            "<div class='training-journal-label'>{stage} / {status}</div>"
            "<div class='training-journal-text'>{message}</div>"
            "<div class='training-journal-meta'>{meta}</div>"
            "</div>".format(
                state=_escape(_log_state(f"{status} {row.get('message', '')}")),
                stage=_escape(stage),
                status=_escape(status),
                message=_escape(row.get("message", "")),
                meta=_escape(", ".join(meta_parts)),
            )
        )

    for note in notes:
        blocks.append(
            "<div class='training-journal-entry'>"
            "<div class='training-journal-label'>Note</div>"
            "<div class='training-journal-text'>{note}</div>"
            "</div>".format(note=_escape(note))
        )

    with st.expander("Training journal", expanded=bool(log_rows)):
        st.markdown("<div class='training-journal'>" + "".join(blocks) + "</div>", unsafe_allow_html=True)


def latest_iteration(result: Dict[str, Any]) -> Dict[str, Any]:
    for item in reversed((result or {}).get("iterations", [])):
        if "error" not in item:
            return item
    return {}


def result_label(result: Dict[str, Any]) -> str:
    item = latest_iteration(result)
    graph = item.get("architect", {}).get("graph", {}) if item else {}
    engineer = item.get("engineer", {}) if item else {}
    strategy_name = engineer.get("industrial_strategy", "") or "default"
    rows = graph_rows(graph) if graph else []
    if not rows:
        return "evaluation"
    model = next((row for row in rows if row["role"] == "model"), rows[-1])
    params = "" if model["params"] == "-" else f" ({model['params']})"
    prefix = f"{strategy_name} + " if strategy_name != "default" else ""
    return f"{prefix}{model['operation']}{params}"


def result_metric_value(result: Dict[str, Any]) -> tuple[str, Any]:
    item = latest_iteration(result)
    engineer = item.get("engineer", {}) if item else {}
    metrics = engineer.get("test_metrics", {}) or engineer.get("graph_metrics", {}) or {}
    metric = metrics.get("primary_metric") or result.get("primary_metric") or "score"
    value = metrics.get("primary_metric_value", engineer.get("graph_score"))
    return str(metric), value


def result_score(result: Dict[str, Any]) -> float | None:
    _, value = result_metric_value(result)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compact_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key, value in (metrics or {}).items():
        if key == "error":
            compact[key] = str(value)[:300]
        elif key in METRIC_METADATA_KEYS:
            compact[key] = value
        else:
            number = _numeric_or_none(value)
            if number is not None:
                compact[key] = number
    return compact


def compact_result_for_agent(result: Dict[str, Any], source: str = "", saved_at: str = "", label: str = "") -> Dict[str, Any]:
    item = latest_iteration(result)
    if not item:
        return {}
    engineer = item.get("engineer", {}) or {}
    architect = item.get("architect", {}) or {}
    critic = item.get("critic", {}) or {}
    metric, value = result_metric_value(result)
    train_metrics = compact_metrics(engineer.get("train_metrics", {}) or {})
    test_metrics = compact_metrics(engineer.get("test_metrics", {}) or engineer.get("graph_metrics", {}) or {})
    graph = architect.get("graph") or engineer.get("assumption_graph") or {}
    return {
        "source": source,
        "saved_at": saved_at,
        "label": label or result_label(result),
        "score": _numeric_or_none(value),
        "primary_metric": metric,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "critic_decision": critic.get("winner", ""),
        "suggested_mutations": critic.get("suggested_mutations", []) or [],
        "graph": graph,
        "summary": (result.get("report", {}) or {}).get("summary", ""),
    }


def previous_evaluations_for_agent() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    current = compact_result_for_agent(
        st.session_state.get("result") or {},
        source="current_before_new_run",
        label="Current before new run",
    )
    if current:
        rows.append(current)
    for entry in st.session_state.get("evaluation_history", []) or []:
        compact = compact_result_for_agent(
            entry.get("result", {}) or {},
            source=entry.get("reason", "previous"),
            saved_at=entry.get("saved_at", ""),
            label=entry.get("label", ""),
        )
        if compact:
            rows.append(compact)
    best = compact_result_for_agent(
        st.session_state.get("best_result") or {},
        source="saved_best",
        saved_at=st.session_state.get("best_saved_at", ""),
        label=st.session_state.get("best_label", "Saved best"),
    )
    if best:
        rows.append(best)
    return rows[:10]


def update_best_result(candidate: Dict[str, Any]) -> None:
    candidate_score = result_score(candidate)
    if candidate_score is None:
        return
    best = st.session_state.get("best_result") or {}
    best_score = result_score(best)
    if best_score is None or candidate_score > best_score:
        st.session_state.best_result = copy.deepcopy(candidate)
        st.session_state.best_saved_at = pd.Timestamp.now().strftime("%H:%M:%S")
        st.session_state.best_label = result_label(candidate)


def best_graph_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    best_eval = result.get("best_evaluation", {}) or {}
    if isinstance(best_eval.get("graph"), dict) and best_eval["graph"]:
        return best_eval["graph"]
    best_result = st.session_state.get("best_result") or {}
    item = latest_iteration(best_result)
    if item:
        return (item.get("architect", {}) or {}).get("graph", {}) or (item.get("engineer", {}) or {}).get("assumption_graph", {})
    return {}


def archive_current_result(reason: str) -> None:
    current = st.session_state.get("result")
    if not current or not latest_iteration(current):
        return
    update_best_result(current)
    history = st.session_state.setdefault("evaluation_history", [])
    history.insert(0, {
        "saved_at": pd.Timestamp.now().strftime("%H:%M:%S"),
        "reason": reason,
        "label": result_label(current),
        "result": copy.deepcopy(current),
    })
    del history[20:]


def metric_rows(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"metric": key, "value": format_number(value)}
        for key, value in (metrics or {}).items()
        if key not in METRIC_METADATA_KEYS and value is not None
    ]


def render_metric_table(title: str, metrics: Dict[str, Any]) -> None:
    st.write(title)
    if not metrics:
        st.caption("No metrics returned for this split.")
        return

    primary_metric = metrics.get("primary_metric")
    primary_value = metrics.get("primary_metric_value")
    if primary_metric and primary_value is not None:
        st.caption(f"Primary metric: {primary_metric} = {format_number(primary_value)}")

    rows = metric_rows(metrics)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No displayable metric values.")


def render_split_info(split_info: Dict[str, Any]) -> None:
    if not split_info:
        return
    test_size = split_info.get("test_size")
    parts = []
    if split_info.get("n_train") is not None:
        parts.append(f"train rows: {split_info['n_train']}")
    if split_info.get("n_test") is not None:
        parts.append(f"test rows: {split_info['n_test']}")
    if test_size is not None:
        parts.append(f"test_size: {format_number(test_size)}")
    if parts:
        st.caption("Hold-out split - " + ", ".join(parts))


def graph_rows(graph: Dict[str, Any]) -> List[Dict[str, str]]:
    root_id = root_node_id(graph)
    rows = []
    for node in graph.get("nodes", []):
        role = "model" if node.get("id") == root_id else "preprocessing"
        rows.append({
            "id": node.get("id", ""),
            "role": role,
            "operation": node.get("operation", ""),
            "inputs": ", ".join(node.get("inputs", [])) or "raw data",
            "params": format_params(node.get("params", {}) or {}),
        })
    return rows


def has_pending_draft() -> bool:
    return bool(st.session_state.get("graph")) and not bool(st.session_state.get("graph_approved"))


def render_pending_draft_panel(key_prefix: str = "pending_draft") -> None:
    if not has_pending_draft():
        return

    draft = st.session_state.get("graph", {}) or {}
    st.info("Architect prepared a revised draft. It is not the active evaluated pipeline until you approve it.")
    render_graph(draft, show_details=False)
    rows = graph_rows(draft)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    approve_col, discard_col = st.columns(2)
    if approve_col.button("Approve Draft", use_container_width=True, key=f"{key_prefix}_approve"):
        approve_graph()
        st.session_state.revision_success_message = "Draft approved. You can now evaluate this pipeline."
        st.rerun()
    if discard_col.button(
        "Discard Draft",
        use_container_width=True,
        disabled=not st.session_state.get("approved_graph"),
        key=f"{key_prefix}_discard",
    ):
        discard_draft()
        st.session_state.revision_success_message = "Draft discarded; approved graph restored."
        st.rerun()


def metric_comparison_rows(engineer: Dict[str, Any]) -> List[Dict[str, Any]]:
    train_metrics = engineer.get("train_metrics", {}) or {}
    test_metrics = engineer.get("test_metrics", {}) or engineer.get("graph_metrics", {}) or {}
    names = [
        name
        for name in dict.fromkeys(list(train_metrics.keys()) + list(test_metrics.keys()))
        if name not in METRIC_METADATA_KEYS
    ]
    rows: List[Dict[str, Any]] = []
    for name in names:
        train_value = _numeric_or_none(train_metrics.get(name))
        test_value = _numeric_or_none(test_metrics.get(name))
        if train_value is None and test_value is None:
            continue
        gap = None
        if train_value is not None and test_value is not None:
            gap = train_value - test_value
        rows.append({
            "metric": name,
            "train": train_value,
            "test": test_value,
            "gap": gap,
        })
    return rows


def render_model_assessment(engineer: Dict[str, Any]) -> None:
    rows = metric_comparison_rows(engineer)
    if not rows:
        st.caption("No train/test metric comparison is available for this run.")
        return

    st.write("Train/Test Assessment")
    chart = pd.DataFrame(rows).set_index("metric")
    st.bar_chart(chart[["train", "test"]])
    st.dataframe(
        pd.DataFrame([
            {
                "metric": row["metric"],
                "train": format_number(row["train"]) if row["train"] is not None else "",
                "test": format_number(row["test"]) if row["test"] is not None else "",
                "gap": format_number(row["gap"]) if row["gap"] is not None else "",
            }
            for row in rows
        ]),
        use_container_width=True,
        hide_index=True,
    )


def render_explain_graph_assessment(explanation: Dict[str, Any]) -> None:
    if not explanation or explanation.get("skipped"):
        st.caption("Graph explanation is unavailable for this run.")
        return

    feature_importance = explanation.get("feature_importance") or {}

    if feature_importance:
        rows = sorted(
            (
                {"feature": str(feature), "importance": _numeric_or_none(value)}
                for feature, value in feature_importance.items()
            ),
            key=lambda row: row["importance"] if row["importance"] is not None else -1.0,
            reverse=True,
        )
        rows = [row for row in rows if row["importance"] is not None]
        if rows:
            st.write("Feature Importance")
            frame = pd.DataFrame(rows).set_index("feature")
            st.bar_chart(frame["importance"])
            st.dataframe(
                pd.DataFrame([
                    {"feature": row["feature"], "importance": format_number(row["importance"])}
                    for row in rows
                ]),
                use_container_width=True,
                hide_index=True,
            )


def render_best_graph_assessment(result: Dict[str, Any]) -> None:
    comparison = result.get("current_vs_best", {}) or (result.get("report", {}) or {}).get("current_vs_best", {}) or {}
    best_eval = result.get("best_evaluation", {}) or (result.get("report", {}) or {}).get("best_evaluation", {}) or {}
    best_graph = best_graph_from_result(result)
    best_score = _numeric_or_none(comparison.get("best_score") or best_eval.get("score"))
    current_score = _numeric_or_none(comparison.get("current_score"))
    metric = comparison.get("primary_metric") or best_eval.get("primary_metric") or result.get("primary_metric") or "score"

    if best_score is None and not best_graph:
        st.caption("No saved best graph is available yet.")
        return

    st.write("Saved Best Graph")
    cols = st.columns(3)
    cols[0].metric(f"Best {metric}", format_number(best_score) if best_score is not None else "-")
    if current_score is not None:
        delta = current_score - best_score if best_score is not None else None
        cols[1].metric(f"Current {metric}", format_number(current_score))
        cols[2].metric("Current vs best", format_number(delta) if delta is not None else "-")
        if delta is not None and delta < 0:
            st.warning(
                f"The latest run is below the saved best by {format_number(abs(delta))} {metric}. "
                "Keep the saved best graph unless there is a non-metric reason to prefer the current run."
            )
    if best_eval.get("label") or best_eval.get("saved_at"):
        st.caption(
            "Best source: "
            + " - ".join(str(part) for part in [best_eval.get("label"), best_eval.get("saved_at")] if part)
        )

    if best_graph:
        render_graph(best_graph, show_details=False)
        if st.button("Restore Saved Best Graph", use_container_width=True, key="restore_saved_best_graph"):
            st.session_state.graph = best_graph
            st.session_state.approved_graph = best_graph
            st.session_state.mermaid = ""
            st.session_state.approved_mermaid = ""
            st.session_state.graph_approved = True
            st.success("Saved best graph restored as the approved graph.")
            st.rerun()


def industrial_strategies_for_task(config: Dict[str, Any], task_type: str) -> List[Dict[str, Any]]:
    catalog = config.get("industrial_strategies") or config.get("training_strategies") or {}
    return list(catalog.get(task_type, []) or [])


def strategy_rows(config: Dict[str, Any], task_type: str) -> List[Dict[str, str]]:
    strategies = industrial_strategies_for_task(config, task_type)
    return [
        {
            "strategy": item.get("name", ""),
            "label": item.get("label", ""),
            "description": item.get("description", ""),
            "source": item.get("fedot_industrial_reference", ""),
            "default params": format_params(item.get("default_params", {}) or {}),
        }
        for item in strategies
    ]


def render_industrial_strategy_summary() -> None:
    name = st.session_state.get("industrial_strategy") or "default"
    params = st.session_state.get("industrial_strategy_params") or {}
    task_type = st.session_state.get("task_type", "classification")
    data_type = "time_series" if task_type in ("ts_classification", "ts_regression", "ts_forecasting") else "table"
    data_type_text = "table (tabular)" if data_type == "table" else data_type
    if name == "default":
        st.caption(
            "Industrial strategy: default "
            f"(Fedot.Industrial default strategy; config data_type={data_type_text})."
        )
        return
    st.caption(f"Industrial strategy: {name}")
    if params:
        st.code(json.dumps(params, ensure_ascii=False, indent=2), language="json")


def render_strategy_runtime_notice() -> None:
    name = st.session_state.get("industrial_strategy") or "default"
    if name != "default":
        st.warning(
            f"Fedot.Industrial strategy '{name}' is selected. It may train several internal "
            "AutoML branches, so evaluation can take noticeably longer than the default path."
        )


def render_strategy_selection_details(strategy_name: str, strategy_spec: Dict[str, Any]) -> None:
    if strategy_name == "default":
        st.info(
            "Default execution: no Fedot.Industrial training strategy is selected. The backend "
            "passes task data_type into the Fedot.Industrial config."
        )
        return

    description = strategy_spec.get("description")
    if description:
        st.caption(description)

    notice = strategy_spec.get("runtime_notice")
    if notice:
        st.warning(notice)

    effects = strategy_spec.get("pipeline_effects", []) or []
    if effects:
        st.write("What changes in the pipeline")
        for effect in effects:
            st.write(f"- {effect}")

    editable = strategy_spec.get("editable_params", []) or []
    if editable:
        st.write("What you can change for this strategy")
        st.dataframe(pd.DataFrame(editable), use_container_width=True, hide_index=True)


def render_industrial_strategy_picker(config: Dict[str, Any], task_type: str, key_suffix: str = "") -> None:
    """Pick the Fedot.Industrial execution strategy applied to the next evaluation."""
    available = industrial_strategies_for_task(config, task_type)
    names = ["default"] + [
        item.get("name", "")
        for item in available
        if item.get("name") and item.get("name") not in ("default", "tabular")
    ]

    current_name = st.session_state.get("industrial_strategy") or "default"
    if current_name not in names:
        current_name = "default"
        st.session_state.industrial_strategy = "default"
    current_index = names.index(current_name) if current_name in names else 0

    widget_key = f"industrial_strategy_select_{key_suffix}"
    if widget_key in st.session_state and st.session_state.get(widget_key) not in names:
        st.session_state[widget_key] = "default"
    selected = st.selectbox(
        "Industrial strategy",
        names,
        index=current_index,
        key=widget_key,
        help=(
            "Default runs Fedot.Industrial's default strategy. federated_automl and "
            "sampling_strategy remain available as alternate Fedot.Industrial strategies."
        ),
    )
    selected_spec = next((item for item in available if item.get("name") == selected), {})
    render_strategy_selection_details(selected, selected_spec)

    default_params = selected_spec.get("default_params", {}) or {}
    if selected == st.session_state.get("industrial_strategy"):
        shown_params = st.session_state.get("industrial_strategy_params") or default_params
    else:
        shown_params = default_params
    if selected == "default":
        shown_params = {}

    params_text = st.text_area(
        "Strategy parameters",
        value=json.dumps(shown_params or {}, ensure_ascii=False, indent=2),
        height=110,
        disabled=selected == "default",
        key=f"industrial_strategy_params_{key_suffix}",
    )
    if st.button("Apply strategy", key=f"industrial_strategy_apply_{key_suffix}"):
        try:
            params = {} if selected == "default" else json.loads(params_text or "{}")
            if not isinstance(params, dict):
                raise ValueError("strategy parameters must be a JSON object")
            st.session_state.industrial_strategy = selected
            st.session_state.industrial_strategy_params = params
            st.success(
                "Industrial strategy applied for the next evaluation: "
                f"{selected}{' with custom params' if params else ''}."
            )
        except (json.JSONDecodeError, ValueError) as exc:
            st.error(f"Invalid strategy parameters: {exc}")


def mutation_rows(mutations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows = []
    for mutation in mutations:
        kind = mutation.get("type", "")
        if kind == "replace":
            rows.append({
                "action": "replace",
                "node": mutation.get("node_id", ""),
                "target": mutation.get("new_operation", ""),
                "details": "",
            })
        elif kind == "add":
            node = mutation.get("node", {}) or {}
            rows.append({
                "action": "add",
                "node": node.get("id", ""),
                "target": node.get("operation", ""),
                "details": f"before {mutation.get('rewire_input_of', '')}".strip(),
            })
        elif kind == "remove":
            rows.append({
                "action": "remove",
                "node": mutation.get("node_id", ""),
                "target": "",
                "details": "",
            })
        elif kind == "connect":
            rows.append({
                "action": "connect",
                "node": mutation.get("node_id", ""),
                "target": mutation.get("input_id", ""),
                "details": "",
            })
        else:
            rows.append({
                "action": kind,
                "node": mutation.get("node_id", ""),
                "target": mutation.get("input_id", ""),
                "details": "",
            })
    return rows


def render_operation_catalog(config: Dict[str, Any], task_type: str, key_suffix: str = "catalog") -> None:
    rows = operation_rows(config, task_type)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        return
    st.info("Operation catalog is unavailable. Check that backend /config is reachable, then refresh metadata.")
    if st.button("Refresh Backend Metadata", key=f"refresh_metadata_{key_suffix}_{task_type}"):
        st.rerun()


def mutate_graph(mutation: Dict[str, Any]) -> None:
    graph = st.session_state.get("graph")
    if not graph:
        st.warning("Ask Architect for a graph first.")
        return
    try:
        result = post_json("/graph/mutate", {"graph": graph, "mutation": mutation})
        if result.get("valid"):
            set_graph(result)
            st.success("Graph updated.")
        else:
            st.error(result.get("message", "Invalid graph mutation"))
            render_diagnostics(result.get("diagnostics", []), "Mutation diagnostics")
    except Exception as exc:
        st.error(f"Mutation failed: {exc}")


def update_node(node_id: str, new_operation: str) -> None:
    # Graph edits are structural only. Fedot.Industrial handles parameter
    # polishing during finetune.
    mutate_graph({
        "type": "replace",
        "node_id": node_id,
        "new_operation": new_operation,
    })


def stream_run(payload: Dict[str, Any]) -> None:
    log_box = st.empty()
    progress = st.progress(0)
    lines: List[str] = []
    seen_diagnostics = set()
    total_steps = 4
    done_steps = 0
    progress_value = 0.0

    with requests.post(f"{BACKEND_URL}/orchestrate/stream", json=payload, stream=True, timeout=1800) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            event_type = event.get("event")
            if event_type == "status":
                message = event.get("message", "")
                if message:
                    lines.append(message)
            elif event_type == "agent_start":
                lines.append(f"{event.get('agent')}: started")
            elif event_type == "agent_progress":
                agent = event.get("agent", "Agent")
                elapsed = event.get("elapsed_seconds", 0)
                message = event.get("message", "still running")
                try:
                    elapsed_seconds = max(0, int(float(elapsed or 0)))
                except (TypeError, ValueError):
                    elapsed_seconds = 0
                lines.append(f"{agent}: {message} ({format_elapsed(elapsed_seconds)})")
                if agent == "Engineer":
                    progress_value = max(
                        progress_value,
                        min(0.70, 0.25 + min(elapsed_seconds, 1800) / 1800 * 0.35),
                    )
                    progress.progress(progress_value)
            elif event_type == "agent_done":
                summary = str(event.get("summary", ""))
                if event.get("agent") == "Critic":
                    summary = summary.replace("needs_revision", "Need revision").replace("accepted", "Accepted")
                lines.append(f"{event.get('agent')}: {summary}")
                if event.get("diagnostics"):
                    for diagnostic in event["diagnostics"]:
                        key = (event.get("agent"), diagnostic.get("kind"), diagnostic.get("summary"))
                        if key not in seen_diagnostics:
                            seen_diagnostics.add(key)
                            lines.append(f"{event.get('agent')} note: {diagnostic.get('summary', '')}")
                done_steps += 1
                progress_value = min(done_steps / total_steps, 1.0)
                progress.progress(progress_value)
            elif event_type == "diagnostics":
                for diagnostic in event.get("diagnostics", []):
                    key = (event.get("agent"), diagnostic.get("kind"), diagnostic.get("summary"))
                    if key not in seen_diagnostics:
                        seen_diagnostics.add(key)
                        lines.append(f"{event.get('agent')} note: {diagnostic.get('summary', '')}")
            elif event_type == "iteration_done":
                primary_metric = event.get("primary_metric") or "score"
                primary_value = event.get("primary_metric_value", event.get("graph_score", 0))
                lines.append(
                    f"Evaluation done: test {primary_metric}={format_number(primary_value)}, "
                    f"critic decision={decision_label(event.get('winner'))}"
                )
                if event.get("industrial_strategy"):
                    st.session_state.industrial_strategy = event.get("industrial_strategy") or "default"
                    st.session_state.industrial_strategy_params = dict(
                        event.get("industrial_strategy_params") or {}
                    )
                if event.get("graph"):
                    st.session_state.graph = event["graph"]
                    st.session_state.approved_graph = event["graph"]
                    st.session_state.mermaid = event.get("mermaid", "")
                    st.session_state.approved_mermaid = event.get("mermaid", "")
                    st.session_state.graph_approved = True
            elif event_type == "error":
                lines.append(f"ERROR: {event.get('message')}")
            elif event_type == "complete":
                archive_current_result("Previous full evaluation")
                completed_result = event.get("result", {})
                st.session_state.result = completed_result
                update_best_result(completed_result)
                progress_value = 1.0
                progress.progress(1.0)
                lines.append("Run completed.")

            with log_box.container():
                render_run_log(lines)


def sidebar(config: Dict[str, Any]) -> None:
    with st.sidebar:
        st.header("Dataset")
        uploaded = st.file_uploader("CSV file", type="csv")
        if uploaded:
            try:
                path = save_uploaded_csv(uploaded)
                st.success(f"Saved: {Path(path).name}")
            except Exception as exc:
                st.error(f"Cannot read CSV: {exc}")

        if "df" in st.session_state:
            df = st.session_state.df
            st.caption(f"{df.shape[0]} rows, {df.shape[1]} columns")
            columns = list(df.columns)
            current_target = st.session_state.get("target_column")
            target_index = columns.index(current_target) if current_target in columns else 0
            st.session_state.target_column = st.selectbox("Target column", columns, index=target_index)
            current_task = st.session_state.get("task_type", "classification")
            task_index = TASK_TYPES.index(current_task) if current_task in TASK_TYPES else 0
            st.session_state.task_type = st.selectbox("Task type", TASK_TYPES, index=task_index)
            metric_options = (
                config.get("metrics_by_task", {}).get(st.session_state.task_type)
                or DEFAULT_METRICS_BY_TASK.get(st.session_state.task_type, [])
            )
            if metric_options:
                current_metric = st.session_state.get("primary_metric")
                default_index = metric_options.index(current_metric) if current_metric in metric_options else 0
                st.session_state.primary_metric = st.selectbox(
                    "Primary metric",
                    metric_options,
                    index=default_index,
                    help="Engineer and Critic will rank the approved graph by this metric.",
                )
            else:
                st.session_state.primary_metric = ""
            if st.session_state.task_type == "ts_forecasting":
                st.session_state.forecast_length = st.number_input("Forecast length", min_value=1, value=14)
            else:
                st.session_state.forecast_length = None

            st.session_state.test_size = st.slider(
                "Test size (hold-out split)",
                min_value=0.05,
                max_value=0.5,
                value=float(st.session_state.get("test_size", 0.2)),
                step=0.05,
                help="Share of the dataset reserved for the test split. Train metrics and test metrics are reported separately.",
            )

            st.markdown("**Industrial strategy**")
            render_industrial_strategy_picker(config, st.session_state.task_type, key_suffix="sidebar")
            render_industrial_strategy_summary()

        st.divider()
        st.caption(f"Backend: {BACKEND_URL}")
        try:
            health = requests.get(f"{BACKEND_URL}/health", timeout=2).json()
            st.success(health.get("status", "connected"))
        except Exception:
            st.error("backend unavailable")
        if st.button("Refresh Backend Metadata", key="sidebar_refresh_metadata"):
            st.rerun()


def data_tab() -> None:
    st.subheader("Data")
    if not require_dataset():
        return
    df = st.session_state.df
    st.write(f"Path: `{st.session_state.csv_path}`")
    st.dataframe(df.head(50), use_container_width=True)
    st.write("Numeric summary")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)


def architect_tab(config: Dict[str, Any]) -> None:
    st.subheader("Architect")
    if not require_dataset():
        return

    task_type = st.session_state.get("task_type", "classification")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.write("Available atomic operations")
        render_operation_catalog(config, task_type, "architect")
        strategy_catalog_rows = strategy_rows(config, task_type)
        if strategy_catalog_rows:
            st.write("Available industrial execution strategies")
            st.dataframe(pd.DataFrame(strategy_catalog_rows), use_container_width=True, hide_index=True)
            st.caption(
                "Strategies are picked outside the graph. Use the sidebar to switch strategy "
                "before evaluating."
            )

        message = st.text_area(
            "Request to Architect",
            placeholder="Example: prefer a simpler linear model, use frequency features for a signal task, or keep the graph conservative.",
            height=110,
        )
        if st.button("Ask Architect", type="primary", use_container_width=True, key="architect_ask"):
            try:
                payload = current_payload(
                    {
                        "message": message,
                        "current_graph": st.session_state.get("graph"),
                    }
                )
                result = post_json("/architect/chat", payload, timeout=180)
                set_graph(result)
                st.session_state.architect_tool_calls = result.get("tool_calls", [])
                st.success("Architect proposed a graph.")
            except Exception as exc:
                st.error(f"Architect failed: {exc}")

    with col_right:
        status = "approved" if st.session_state.get("graph_approved") else "draft"
        st.caption(f"Graph status: {status}")
        render_graph(st.session_state.get("graph", {}))
        if st.session_state.get("architect_analysis"):
            st.write("Analysis")
            st.write(st.session_state.architect_analysis)
        if st.session_state.get("architect_reasoning"):
            st.write("Reasoning")
            st.write(st.session_state.architect_reasoning)
        render_diagnostics(st.session_state.get("architect_diagnostics", []), "Architect diagnostics")
        if st.session_state.get("graph"):
            action_cols = st.columns(2)
            if action_cols[0].button("Approve Draft", use_container_width=True, key="architect_approve_draft"):
                approve_graph()
                st.success("Graph approved.")
            if action_cols[1].button("Discard Draft", use_container_width=True, disabled=not st.session_state.get("approved_graph"), key="architect_discard_draft"):
                discard_draft()
                st.info("Draft discarded; approved graph restored.")
        render_tool_calls(st.session_state.get("architect_tool_calls", []), "Architect tool calls")


def graph_editor_tab(config: Dict[str, Any]) -> None:
    st.subheader("Graph Editor")
    graph = st.session_state.get("graph", {})
    task_type = (
        graph.get("task_type", st.session_state.get("task_type", "classification"))
        if graph
        else st.session_state.get("task_type", "classification")
    )
    operations = config.get("operations", {}).get(task_type, {})
    preprocessing_ops = operations.get("preprocessing", [])
    model_ops = operations.get("models", [])

    if not graph:
        st.info("Ask Architect for a graph, or create a draft manually here.")
        if model_ops:
            with st.form("create_graph_draft"):
                prep_options = ["(none)"] + list(preprocessing_ops or [])
                prep_op = st.selectbox("Preprocessing", prep_options, key="create_graph_preprocessing")
                model_op = st.selectbox("Model", model_ops, key="create_graph_model")
                create_clicked = st.form_submit_button("Create draft graph")
                if create_clicked:
                    nodes: List[Dict[str, Any]] = []
                    model_inputs: List[str] = []
                    if prep_op != "(none)":
                        nodes.append({
                            "id": "preprocessing",
                            "operation": prep_op,
                            "params": {},
                            "inputs": [],
                        })
                        model_inputs = ["preprocessing"]
                    nodes.append({
                        "id": "model",
                        "operation": model_op,
                        "params": {},
                        "inputs": model_inputs,
                    })
                    adopt_graph_draft({"task_type": task_type, "nodes": nodes})
                    st.success("Draft graph created.")
                    st.rerun()
        else:
            st.warning("No model operations are exposed for this task.")
        st.write("Available atomic operations")
        render_operation_catalog(config, task_type, "editor_empty")
        return

    root_id = root_node_id(graph)
    nodes = graph.get("nodes", [])
    model_op_set = set(model_ops or [])
    preprocessing_op_set = set(preprocessing_ops or [])
    current_model = next(
        (
            node
            for node in nodes
            if node.get("id") == root_id
            and (
                node.get("operation") in model_op_set
                or node.get("operation") not in preprocessing_op_set
            )
        ),
        {},
    )
    current_model_id = current_model.get("id", "")
    preprocessing_nodes = [node for node in nodes if node.get("id") != current_model_id]

    status_col, approve_col, discard_col = st.columns([2, 1, 1])
    status_col.caption("Approved" if st.session_state.get("graph_approved") else "Draft")
    if approve_col.button("Approve Edited Graph", use_container_width=True, key="editor_approve_graph"):
        approve_graph()
        st.success("Graph approved.")
    if discard_col.button("Discard Draft", use_container_width=True, disabled=not st.session_state.get("approved_graph"), key="editor_discard_draft"):
        discard_draft()
        st.info("Draft discarded; approved graph restored.")

    overview_tab, edit_tab, operations_tab, evaluate_tab = st.tabs(["Overview", "Edit", "Operations", "Evaluate"])

    with overview_tab:
        graph_col, table_col = st.columns([1, 1])
        with graph_col:
            render_graph(graph, show_details=False)
        with table_col:
            st.dataframe(pd.DataFrame(graph_rows(graph)), use_container_width=True, hide_index=True)
            if st.session_state.get("architect_analysis"):
                st.write("Analysis")
                st.write(st.session_state.architect_analysis)
            if st.session_state.get("architect_reasoning"):
                st.write("Reasoning")
                st.write(st.session_state.architect_reasoning)

    with edit_tab:
        preprocessing_col, model_col = st.columns([1, 1])

        with preprocessing_col:
            st.write("Preprocessing")
            if preprocessing_nodes:
                selected_options = [node.get("id", "") for node in preprocessing_nodes]
                selected_id = st.selectbox("Node", selected_options, key="preprocessing_editor_node")
                current_preprocessing = next(
                    (node for node in preprocessing_nodes if node.get("id") == selected_id),
                    preprocessing_nodes[0],
                )
                available_preprocessing_ops = preprocessing_ops or [current_preprocessing.get("operation", "")]
                if current_preprocessing.get("operation") not in available_preprocessing_ops:
                    available_preprocessing_ops = [current_preprocessing.get("operation", "")] + available_preprocessing_ops
                with st.form("edit_preprocessing"):
                    prep_index = available_preprocessing_ops.index(current_preprocessing.get("operation"))
                    prep_op = st.selectbox("Operation", available_preprocessing_ops, index=prep_index)
                    apply_prep, remove_prep = st.columns([1, 1])
                    apply_clicked = apply_prep.form_submit_button("Apply")
                    remove_clicked = remove_prep.form_submit_button("Remove")
                    if apply_clicked:
                        update_node(current_preprocessing.get("id", ""), prep_op)
                    if remove_clicked:
                        mutate_graph({"type": "remove", "node_id": current_preprocessing.get("id", "")})
            elif preprocessing_ops and current_model:
                root_inputs = current_model.get("inputs", [])
                with st.form("add_preprocessing"):
                    new_id = st.text_input("Node id", value="preprocessing")
                    prep_op = st.selectbox("Operation", preprocessing_ops, key="add_preprocessing_op")
                    add_clicked = st.form_submit_button("Add preprocessing")
                    if add_clicked:
                        mutate_graph({
                            "type": "add",
                            "node": {
                                "id": new_id,
                                "operation": prep_op,
                                "inputs": root_inputs,
                            },
                            "rewire_input_of": root_id,
                        })
            elif preprocessing_ops:
                st.caption("Add a model first, then insert preprocessing before it.")
            else:
                st.caption("No preprocessing operations are exposed for this task.")

        with model_col:
            st.write("Model")
            if model_ops and current_model:
                available_model_ops = model_ops
                if current_model.get("operation") not in available_model_ops:
                    available_model_ops = [current_model.get("operation", "")] + available_model_ops
                with st.form("edit_model"):
                    st.text_input("Node id", value=root_id, disabled=True)
                    model_index = available_model_ops.index(current_model.get("operation"))
                    model_op = st.selectbox("Operation", available_model_ops, index=model_index, key="model_editor_op")
                    submitted = st.form_submit_button("Apply model")
                    if submitted:
                        update_node(root_id, model_op)
            elif model_ops:
                existing_ids = [node.get("id", "") for node in nodes]
                default_model_id = "model" if "model" not in existing_ids else "model_new"
                input_options = [node_id for node_id in existing_ids if node_id]
                default_inputs = [node_id for node_id in root_node_ids(graph) if node_id in input_options]
                with st.form("add_model"):
                    new_id = st.text_input("Node id", value=default_model_id, key="add_model_id")
                    model_op = st.selectbox("Operation", model_ops, key="add_model_op")
                    selected_inputs = st.multiselect(
                        "Inputs",
                        input_options,
                        default=default_inputs,
                        key="add_model_inputs",
                    )
                    submitted = st.form_submit_button("Add model")
                    if submitted:
                        mutate_graph({
                            "type": "add",
                            "node": {
                                "id": new_id,
                                "operation": model_op,
                                "inputs": selected_inputs,
                            },
                        })
            else:
                st.caption("No model operations returned by backend config.")

        with st.expander("Industrial execution strategy"):
            st.caption(
                "Industrial strategies are execution modes for Fedot.Industrial. "
                "They are not part of the graph; the graph you approve becomes the assumption "
                "polished by AutoML finetune. Pick a strategy here for the next evaluation."
            )
            render_industrial_strategy_picker(config, task_type, key_suffix="editor")

    with operations_tab:
        render_operation_catalog(config, task_type, "editor")
        strategy_catalog_rows = strategy_rows(config, task_type)
        if strategy_catalog_rows:
            st.write("Training strategies")
            st.dataframe(pd.DataFrame(strategy_catalog_rows), use_container_width=True, hide_index=True)
        strategy_coverage = config.get("fedot_industrial_strategy_catalog", []) or []
        if strategy_coverage:
            st.write("Fedot.Industrial strategy coverage")
            st.dataframe(pd.DataFrame(strategy_coverage), use_container_width=True, hide_index=True)

    with evaluate_tab:
        evaluation_panel()


def evaluation_panel() -> None:
    st.markdown("### Evaluate Approved Graph")
    if not require_dataset():
        return
    approved_graph = st.session_state.get("approved_graph")
    draft_graph = st.session_state.get("graph")
    if not approved_graph:
        st.info("Approve a graph first. Draft graphs are never evaluated automatically.")
        return

    if not st.session_state.get("graph_approved"):
        st.warning("There is an unapproved draft. Evaluation will use the last approved graph.")
        if draft_graph:
            st.write("Draft waiting for approval")
            render_graph(draft_graph, show_details=False)

    st.write("Approved graph for evaluation")
    render_graph(approved_graph, show_details=True)
    render_industrial_strategy_summary()
    render_strategy_runtime_notice()
    if st.button("Evaluate Approved Graph", type="primary", use_container_width=True, key="editor_evaluate_approved_graph"):
        payload = current_payload(
            {
                "iterations": 1,
                "initial_graph": approved_graph,
                "previous_evaluations": previous_evaluations_for_agent(),
            }
        )
        try:
            stream_run(payload)
            if st.session_state.get("result"):
                st.success("MAFIS run completed.")
        except Exception as exc:
            st.error(f"Run failed: {exc}")


def render_engineer_report(
    engineer: Dict[str, Any],
    show_failure_details: bool = True,
    show_training_notes: bool = True,
    show_diagnostics: bool = True,
) -> None:
    st.markdown("#### Engineer Report")
    target_info = engineer.get("target_info", {}) or {}
    if target_info:
        cols = st.columns(4)
        cols[0].metric("Target", target_info.get("column", ""))
        cols[1].metric("Raw dtype", target_info.get("raw_dtype", ""))
        cols[2].metric("Unique", target_info.get("unique_values", 0))
        cols[3].metric("Reference mapping", "yes" if target_info.get("reference_encoded") else "no")
        st.caption(
            "Fedot graph receives raw target values. Mapping is shown only to make class labels readable in diagnostics."
        )
        if target_info.get("reference_encoding"):
            with st.expander("Reference label mapping"):
                mapping_rows = [
                    {"label": label, "code": code}
                    for label, code in target_info["reference_encoding"].items()
                ]
                st.dataframe(pd.DataFrame(mapping_rows), use_container_width=True, hide_index=True)
        if target_info.get("sample_values"):
            st.caption("Target sample: " + ", ".join(target_info["sample_values"][:8]))

    if show_training_notes:
        render_training_journal(engineer)

    if engineer.get("graph_error") and show_failure_details:
        st.error(engineer["graph_error"])

    train_metrics = engineer.get("train_metrics", {}) or {}
    test_metrics = engineer.get("test_metrics", {}) or {}
    has_metrics = bool(train_metrics or test_metrics or engineer.get("graph_metrics"))
    has_error = bool(engineer.get("graph_error"))

    if has_error and not has_metrics:
        st.caption("Train/test metrics are unavailable because the graph did not finish training.")

    if has_metrics and (train_metrics or test_metrics):
        train_col, test_col = st.columns(2)
        with train_col:
            render_metric_table("Train metrics", train_metrics)
        with test_col:
            render_metric_table("Test metrics (hold-out)", test_metrics)
    elif has_metrics:
        metrics = engineer.get("graph_metrics", {}) or {}
        if metrics:
            render_metric_table("Test metrics (hold-out)", metrics)

    if show_diagnostics:
        render_diagnostics(engineer.get("diagnostics", []), "Engineer diagnostics", use_expander=False)


def render_critic_feedback(critic: Dict[str, Any], compact_runtime_error: bool = False, iteration=None) -> None:
    st.markdown("#### Critic Recovery" if compact_runtime_error else "#### Critic Feedback")
    if compact_runtime_error:
        st.caption("Critic is focused on recovery because Engineer reported a training/runtime issue.")
    else:
        st.write(critic.get("assessment", "No critic assessment."))

    cols = st.columns(2)
    cols[0].metric("Decision", decision_label(critic.get("winner", "")))
    cols[1].metric("Suggested changes", len(critic.get("suggested_mutations", []) or []))

    if critic.get("strengths") and not compact_runtime_error:
        st.write("What works")
        for item in critic["strengths"]:
            st.write(f"- {item}")
    if critic.get("weaknesses") and not compact_runtime_error:
        st.write("What should improve")
        for item in critic["weaknesses"]:
            st.write(f"- {item}")
    if critic.get("improvement_plan"):
        st.write("Recovery plan" if compact_runtime_error else "Improvement plan")
        for item in critic["improvement_plan"]:
            st.write(f"- {item}")
    if critic.get("suggested_mutations"):
        recovery_remove_mode = compact_runtime_error or any(
            isinstance(item, dict)
            and item.get("kind") in {
                "node_skipped_after_runtime_error",
                "node_skip_recovery_failed",
                "finetune_fallback_after_node_skip",
            }
            for item in (critic.get("diagnostics", []) or [])
        )
        if recovery_remove_mode:
            st.write("Concrete recovery plan for Architect")
        else:
            st.write("Concrete mutation plan for Architect")
        rows = mutation_rows(critic["suggested_mutations"])
        for idx, (mutation, row) in enumerate(zip(critic["suggested_mutations"], rows)):
            label = f"**{row['action']}** `{row['node']}`"
            if row.get("target"):
                label += f" -> `{row['target']}`"
            if row.get("details"):
                label += f" - {row['details']}"
            default_selected = True
            st.checkbox(label, value=default_selected, key=f"critic_mut_{idx}")
        if iteration is not None:
            if st.button(
                "Apply Selected Mutations",
                type="primary",
                use_container_width=True,
                key="critic_apply_selected_mutations",
            ):
                if selected_critic_mutations(critic):
                    try:
                        request_architect_revision(iteration)
                    except Exception as exc:
                        st.error(f"Could not apply Critic mutations: {exc}")
                else:
                    st.warning("Select at least one mutation to apply.")


def selected_critic_mutations(critic: Dict[str, Any]) -> List[Dict[str, Any]]:
    mutations = critic.get("suggested_mutations", []) or []
    selected = []
    for idx, mutation in enumerate(mutations):
        if st.session_state.get(f"critic_mut_{idx}"):
            selected.append(mutation)
    return selected


def request_architect_revision(iteration: Dict[str, Any], message: str = "") -> None:
    current_graph = st.session_state.get("approved_graph") or iteration.get("architect", {}).get("graph", {})
    critic = iteration.get("critic", {})
    selected = selected_critic_mutations(critic)
    payload = current_payload({
        "current_graph": current_graph,
        "critic_feedback": critic,
        "message": message,
        "selected_mutations": selected,
    })
    result = post_json("/architect/revise", payload, timeout=180)
    set_graph(result, approved=False)
    st.session_state.architect_tool_calls = result.get("tool_calls", [])

    new_strategy = result.get("industrial_strategy")
    if new_strategy:
        st.session_state.industrial_strategy = new_strategy
        st.session_state.industrial_strategy_params = dict(result.get("industrial_strategy_params") or {})

    graph_changes = [m for m in selected if isinstance(m, dict)]
    if graph_changes:
        st.session_state.revision_success_message = (
            f"Architect prepared a new draft from {len(graph_changes)} graph mutation(s). "
            "Review it below or in Graph Editor, then approve it to make it the active pipeline."
        )
    else:
        st.session_state.revision_success_message = (
            "Architect prepared a new draft (no mutations selected; LLM-driven revision). "
            "Review it below or in Graph Editor, then approve it to make it the active pipeline."
        )
    st.rerun()


def _numeric_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluation_summary_row(kind: str, result: Dict[str, Any], current_metric: str, current_value: Any) -> Dict[str, Any]:
    item = latest_iteration(result)
    engineer = item.get("engineer", {}) if item else {}
    critic = item.get("critic", {}) if item else {}
    metric, value = result_metric_value(result)
    train_metrics = engineer.get("train_metrics", {}) or {}
    train_value = train_metrics.get("primary_metric_value")
    current_float = _numeric_or_none(current_value)
    value_float = _numeric_or_none(value)
    delta = ""
    if kind != "current" and metric == current_metric and current_float is not None and value_float is not None:
        delta = format_number(current_float - value_float)
    return {
        "kind": kind,
        "graph": result_label(result),
        "metric": metric,
        "test value": format_number(value),
        "train value": format_number(train_value) if train_value is not None else "",
        "delta to current": delta,
        "critic": decision_label(critic.get("winner", "")),
        "test_size": (engineer.get("split_info", {}) or {}).get("test_size", ""),
    }


def render_evaluation_history(current_result: Dict[str, Any]) -> None:
    history = st.session_state.get("evaluation_history", []) or []
    st.markdown("#### Evaluation History")
    if not history:
        st.caption("No previous evaluations yet. Run another approved graph to compare changes here.")
        return

    current_metric, current_value = result_metric_value(current_result)
    rows = [evaluation_summary_row("current", current_result, current_metric, current_value)]
    for entry in history:
        row = evaluation_summary_row(entry.get("reason", "previous"), entry.get("result", {}), current_metric, current_value)
        row["saved_at"] = entry.get("saved_at", "")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Inspect Previous Evaluation"):
        options = [
            f"{idx + 1}. {entry.get('saved_at', '')} - {entry.get('label', 'evaluation')}"
            for idx, entry in enumerate(history)
        ]
        selected = st.selectbox("Previous run", options, key="history_selected_run")
        selected_entry = history[options.index(selected)]
        previous_result = selected_entry.get("result", {})
        previous_item = latest_iteration(previous_result)
        st.write(selected_entry.get("reason", "previous"))
        render_graph(previous_item.get("architect", {}).get("graph", {}), show_details=False)
        previous_engineer = previous_item.get("engineer", {}) or {}
        render_split_info(previous_engineer.get("split_info", {}) or {})
        train_col, test_col = st.columns(2)
        with train_col:
            render_metric_table("Previous train metrics", previous_engineer.get("train_metrics", {}) or {})
        with test_col:
            render_metric_table(
                "Previous test metrics (hold-out)",
                previous_engineer.get("test_metrics", {}) or previous_engineer.get("graph_metrics", {}) or {},
            )


def adopt_m4_dataset(result: Dict[str, Any]) -> None:
    """Promote a loaded M4 CSV to the active dataset (same shape as CSV upload)."""
    csv_path = result.get("csv_path")
    if not csv_path:
        raise RuntimeError("M4 loader did not return a CSV path.")

    local_path = Path(csv_path)
    if not local_path.exists():
        local_path = shared_data_dir() / Path(csv_path).name
    if not local_path.exists():
        raise FileNotFoundError(f"Loaded M4 CSV not found at {csv_path}.")

    df = pd.read_csv(local_path)
    st.session_state.df = df
    st.session_state.csv_path = csv_path if Path("/app/data").exists() else str(local_path)
    st.session_state.uploaded_signature = f"m4:{local_path.name}"
    st.session_state.target_column = result.get("target_column", "frequency_group")
    st.session_state.task_type = result.get("task_type", "ts_classification")
    st.session_state.forecast_length = None
    st.session_state.industrial_strategy = "default"
    st.session_state.industrial_strategy_params = {}
    st.session_state.result = {}
    st.session_state.evaluation_history = []
    st.session_state.graph = {}
    st.session_state.approved_graph = None
    st.session_state.graph_approved = False
    st.session_state.m4_dataset_info = result


def render_m4_dataset_info(info: Dict[str, Any]) -> None:
    if not info:
        return
    st.markdown("#### Loaded M4 Dataset")
    cols = st.columns(4)
    cols[0].metric("Samples", info.get("n_samples", 0))
    cols[1].metric("Window", info.get("window_length", 0))
    cols[2].metric("Groups", len(info.get("groups", []) or []))
    cols[3].metric("Per group", info.get("n_per_group", 0))
    st.caption(
        f"CSV: `{info.get('csv_path', '')}` - target column: `{info.get('target_column', '')}`. "
        "Open the Data, Architect, Graph Editor and Feedback tabs to run the standard pipeline."
    )

    balance = info.get("class_balance") or {}
    if balance:
        st.write("Class balance")
        st.dataframe(
            pd.DataFrame([{"group": group, "rows": count} for group, count in balance.items()]),
            use_container_width=True,
            hide_index=True,
        )

    source_files = info.get("source_files") or []
    if source_files:
        with st.expander("M4 source files"):
            st.dataframe(pd.DataFrame(source_files), use_container_width=True, hide_index=True)


def benchmarks_tab(config: Dict[str, Any]) -> None:
    st.subheader("Benchmarks")
    bench_config = (config.get("benchmarks", {}) or {}).get("m4_classification", {})
    groups = bench_config.get("groups") or DEFAULT_M4_GROUPS

    st.write("M4 frequency-group classification")
    st.caption(
        "Loads M4 train CSVs through datasetsforecast and saves a fixed-window classification CSV "
        "into the shared data directory. After loading, the rest of the pipeline is exactly the same "
        "as for any uploaded CSV: pick a target, ask Architect for a graph, approve it, evaluate."
    )

    selected_groups = st.multiselect(
        "M4 groups",
        groups,
        default=groups,
        help="At least two groups are recommended because the benchmark is classification by frequency group.",
    )
    cols = st.columns(3)
    n_per_group = cols[0].number_input(
        "Series per group",
        min_value=10,
        max_value=1000,
        value=int(bench_config.get("default_n_per_group", 100)),
        step=10,
    )
    window_length = cols[1].number_input(
        "Window length",
        min_value=8,
        max_value=500,
        value=int(bench_config.get("default_window_length", 50)),
        step=5,
    )
    standardize = cols[2].checkbox("Standardize each series", value=True)

    if len(selected_groups or []) < 2:
        st.warning("Select at least two M4 groups for a classification benchmark.")

    can_load = len(selected_groups or []) >= 2
    if st.button(
        "Load M4 Dataset",
        type="primary",
        use_container_width=True,
        key="m4_load_dataset",
        disabled=not can_load,
    ):
        payload = {
            "groups": selected_groups or groups,
            "n_per_group": int(n_per_group),
            "window_length": int(window_length),
            "standardize": bool(standardize),
        }
        try:
            with st.spinner("Downloading/loading M4 and saving CSV..."):
                result = post_json("/benchmarks/m4/load", payload, timeout=900)
            adopt_m4_dataset(result)
            st.session_state.m4_load_success = (
                f"M4 dataset loaded: {result.get('n_samples', 0)} samples written to "
                f"`{result.get('csv_filename', '')}`. Task type set to "
                f"`{result.get('task_type', 'ts_classification')}`. "
                "Switch to Architect or Graph Editor to continue."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"M4 loading failed: {exc}")

    success_message = st.session_state.pop("m4_load_success", "")
    if success_message:
        st.success(success_message)

    render_m4_dataset_info(st.session_state.get("m4_dataset_info") or {})

    st.divider()
    st.write("Available ts_classification operations")
    render_operation_catalog(config, "ts_classification", "m4_benchmark")


def results_tab() -> None:
    st.subheader("Evaluation Result")
    result = st.session_state.get("result")
    if not result:
        st.info("Evaluate an approved graph to see Engineer and Critic feedback.")
        return

    revision_message = st.session_state.get("revision_success_message", "")
    if revision_message:
        st.success(revision_message)

    item = latest_iteration(result)
    if not item:
        st.error("No successful evaluation found.")
        return

    engineer = item.get("engineer", {})
    critic = item.get("critic", {})

    metrics = engineer.get("test_metrics", {}) or engineer.get("graph_metrics", {}) or {}
    primary_metric = metrics.get("primary_metric") or result.get("primary_metric") or "score"
    primary_value = metrics.get("primary_metric_value", engineer.get("graph_score", 0))
    fallback_used = engineer.get("fallback_used", "")
    finetune_error = engineer.get("finetune_error", "")
    has_runtime_failure = bool(engineer.get("graph_error")) or bool(finetune_error)

    # Header summary: always visible.
    col1, col2, col3 = st.columns(3)
    col1.metric(
        f"Test {primary_metric}",
        format_number(primary_value),
        delta=("(fallback)" if fallback_used else None),
    )
    col2.metric("Critic decision", decision_label(critic.get("winner", "")))
    col3.metric("Suggested changes", len(critic.get("suggested_mutations", []) or []))

    if has_runtime_failure:
        # Show the error banner up front so it's the first thing the user sees.
        render_runtime_failure_once(item)

    # Split the long page into focused sub-tabs so the user does not have to
    # scroll through everything at once.
    sub_tabs = st.tabs([
        "Engineer",
        "Critic",
        "Diagnostics",
        "History",
    ])

    with sub_tabs[0]:
        st.markdown("#### Evaluated Graph")
        render_graph(item.get("architect", {}).get("graph", {}), show_details=False)
        render_engineer_report(
            engineer,
            # When the runtime failure banner is already shown above, suppress
            # the duplicate inline error inside the Engineer report.
            show_failure_details=not has_runtime_failure,
            show_training_notes=True,
            show_diagnostics=False,
        )

    with sub_tabs[1]:
        render_critic_feedback(critic, compact_runtime_error=has_runtime_failure, iteration=item)
        if has_pending_draft():
            st.divider()
            render_pending_draft_panel("critic_pending_draft")

    with sub_tabs[2]:
        eng_diag = engineer.get("diagnostics", []) or []
        critic_diag = critic.get("diagnostics", []) or []
        if not eng_diag and not critic_diag:
            st.caption("No diagnostics produced for this run.")
        else:
            if eng_diag:
                st.markdown("**Engineer diagnostics**")
                render_diagnostics(eng_diag, "Engineer diagnostics", use_expander=False)
            if critic_diag:
                st.markdown("**Critic diagnostics**")
                render_diagnostics(critic_diag, "Critic diagnostics", use_expander=False)

    with sub_tabs[3]:
        render_evaluation_history(result)


def report_tab() -> None:
    st.subheader("Report")
    result = st.session_state.get("result")
    if not result:
        st.info("No report yet.")
        return

    item = latest_iteration(result)
    critic = item.get("critic", {}) if item else {}
    report = result.get("report", {})
    st.markdown(f"### {report.get('title', 'MAFIS report')}")
    st.write(report.get("summary", ""))
    if report.get("methodology"):
        st.write("Methodology")
        st.write(report["methodology"])
    if report.get("results"):
        st.write("Results")
        st.write(report["results"])
    if report.get("recommendations"):
        st.write("Recommendations")
        for recommendation in report["recommendations"]:
            st.write(f"- {recommendation}")

    st.divider()
    st.markdown("#### Visual Assessment")
    render_best_graph_assessment(result)
    st.divider()
    render_explain_graph_assessment(critic.get("explanation", {}) or {})


def tools_tab(tools: Dict[str, Any]) -> None:
    st.subheader("MCP Tools")
    if not tools:
        st.info("Tool metadata is unavailable.")
        return
    for agent, agent_tools in tools.items():
        st.write(agent.title())
        st.dataframe(pd.DataFrame(agent_tools), use_container_width=True, hide_index=True)


def main() -> None:
    st.title(APP_SHORT_NAME)
    st.caption(f"{APP_NAME}: LLM agents compose, train, validate and report Fedot.Industrial pipeline graphs and strategies through MCP tools.")
    if "industrial_strategy" not in st.session_state:
        st.session_state.industrial_strategy = "default"
    if "industrial_strategy_params" not in st.session_state:
        st.session_state.industrial_strategy_params = {}
    config = load_config()
    sidebar(config)
    tools = load_tools()

    tabs = st.tabs(["Data", "Benchmarks", "Architect", "Graph Editor", "Feedback", "Report", "MCP Tools"])
    with tabs[0]:
        data_tab()
    with tabs[1]:
        benchmarks_tab(config)
    with tabs[2]:
        architect_tab(config)
    with tabs[3]:
        graph_editor_tab(config)
    with tabs[4]:
        results_tab()
    with tabs[5]:
        report_tab()
    with tabs[6]:
        tools_tab(tools)


if __name__ == "__main__":
    main()
