import json
import os
import copy
import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="GraphAutoML", layout="wide")

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


def discard_draft() -> None:
    approved = st.session_state.get("approved_graph")
    if approved:
        st.session_state.graph = approved
        st.session_state.mermaid = st.session_state.get("approved_mermaid", "")
        st.session_state.graph_approved = True


def root_node_id(graph: Dict[str, Any]) -> str:
    nodes = graph.get("nodes", [])
    ids = {node.get("id") for node in nodes}
    children = {inp for node in nodes for inp in node.get("inputs", [])}
    roots = [node_id for node_id in ids - children if node_id]
    return roots[0] if roots else ""


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
                for rec in recommendations:
                    st.write(f"- {rec}")
            if item.get("problem_nodes"):
                st.write("Problem nodes")
                st.dataframe(pd.DataFrame(item["problem_nodes"]), use_container_width=True, hide_index=True)
            if item.get("technical_message"):
                st.caption("Technical message")
                st.code(str(item["technical_message"])[:4000])


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


def latest_iteration(result: Dict[str, Any]) -> Dict[str, Any]:
    for item in reversed((result or {}).get("iterations", [])):
        if "error" not in item:
            return item
    return {}


def result_label(result: Dict[str, Any]) -> str:
    item = latest_iteration(result)
    graph = item.get("architect", {}).get("graph", {}) if item else {}
    rows = graph_rows(graph) if graph else []
    if not rows:
        return "evaluation"
    model = next((row for row in rows if row["role"] == "model"), rows[-1])
    params = "" if model["params"] == "-" else f" ({model['params']})"
    return f"{model['operation']}{params}"


def result_metric_value(result: Dict[str, Any]) -> tuple[str, Any]:
    item = latest_iteration(result)
    engineer = item.get("engineer", {}) if item else {}
    metrics = engineer.get("test_metrics", {}) or engineer.get("graph_metrics", {}) or {}
    metric = metrics.get("primary_metric") or result.get("primary_metric") or "score"
    value = metrics.get("primary_metric_value", engineer.get("graph_score"))
    return str(metric), value


def archive_current_result(reason: str) -> None:
    current = st.session_state.get("result")
    if not current or not latest_iteration(current):
        return
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


def mutation_rows(mutations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows = []
    for mutation in mutations:
        kind = mutation.get("type", "")
        if kind == "set_params":
            rows.append({
                "action": "set params",
                "node": mutation.get("node_id", ""),
                "target": "",
                "details": format_params(mutation.get("params", {}) or {}),
            })
        elif kind == "replace":
            rows.append({
                "action": "replace",
                "node": mutation.get("node_id", ""),
                "target": mutation.get("new_operation", ""),
                "details": format_params(mutation.get("params", {}) or {}),
            })
        elif kind == "add":
            node = mutation.get("node", {}) or {}
            rows.append({
                "action": "add",
                "node": node.get("id", ""),
                "target": node.get("operation", ""),
                "details": f"before {mutation.get('rewire_input_of', '')}".strip(),
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


def stream_run(payload: Dict[str, Any]) -> None:
    log_box = st.empty()
    progress = st.progress(0)
    lines: List[str] = []
    seen_diagnostics = set()
    total_steps = 4
    done_steps = 0

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
            elif event_type == "agent_done":
                lines.append(f"{event.get('agent')}: {event.get('summary', '')}")
                if event.get("diagnostics"):
                    for diagnostic in event["diagnostics"]:
                        key = (event.get("agent"), diagnostic.get("kind"), diagnostic.get("summary"))
                        if key not in seen_diagnostics:
                            seen_diagnostics.add(key)
                            lines.append(f"{event.get('agent')} note: {diagnostic.get('summary', '')}")
                done_steps += 1
                progress.progress(min(done_steps / total_steps, 1.0))
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
                    f"critic decision={event.get('winner')}"
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
                st.session_state.result = event.get("result", {})
                progress.progress(1.0)
                lines.append("Run completed.")

            log_box.markdown("\n\n".join(f"- {line}" for line in lines[-14:]))


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
            st.session_state.target_column = st.selectbox("Target column", df.columns)
            st.session_state.task_type = st.selectbox("Task type", TASK_TYPES)
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

        message = st.text_area(
            "Request to Architect",
            placeholder="Example: prefer a simpler linear model, use frequency features for a signal task, or keep the graph conservative.",
            height=110,
        )
        ask_col, default_col = st.columns(2)
        if ask_col.button("Ask Architect", type="primary", use_container_width=True, key="architect_ask"):
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

        if default_col.button("Use Reliable Default", use_container_width=True, key="architect_use_default"):
            graph = {
                "task_type": task_type,
                "nodes": config.get("default_graphs", {}).get(task_type, []),
            }
            set_graph(
                {
                    "graph": graph,
                    "analysis": "Reliable default graph selected without an LLM call.",
                    "reasoning": "This graph is intentionally small so Engineer can first verify that the data format and target are trainable.",
                    "diagnostics": [],
                }
            )
            st.session_state.architect_tool_calls = []
            st.success("Default graph selected.")

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
    if not graph:
        st.info("Ask Architect for a graph, then edit it here.")
        st.write("Available atomic operations")
        render_operation_catalog(config, st.session_state.get("task_type", "classification"), "editor_empty")
        return

    task_type = graph.get("task_type", st.session_state.get("task_type", "classification"))
    operations = config.get("operations", {}).get(task_type, {})
    preprocessing_ops = operations.get("preprocessing", [])
    model_ops = operations.get("models", [])
    node_ids = [node.get("id") for node in graph.get("nodes", [])]
    root_id = root_node_id(graph)

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
        model_col, params_col = st.columns([1, 1])
        with model_col:
            st.write("Model")
            if model_ops:
                with st.form("replace_model"):
                    current_model = next((node for node in graph.get("nodes", []) if node.get("id") == root_id), {})
                    st.text_input("Model node", value=root_id, disabled=True)
                    default_index = model_ops.index(current_model.get("operation")) if current_model.get("operation") in model_ops else 0
                    new_op = st.selectbox("Model operation", model_ops, index=default_index, key="model_op")
                    replace = st.form_submit_button("Replace Model")
                    if replace:
                        mutate_graph({"type": "replace", "node_id": root_id, "new_operation": new_op})
            else:
                st.caption("No model operations returned by backend config.")

        with params_col:
            st.write("Parameters")
            with st.form("params_node"):
                node_id = st.selectbox("Node", node_ids, key="params_node_id")
                current = next((node for node in graph.get("nodes", []) if node.get("id") == node_id), {})
                params_text = st.text_area(
                    "Parameter object",
                    value=json.dumps(current.get("params", {}) or {}, ensure_ascii=False, indent=2),
                    height=130,
                )
                submitted = st.form_submit_button("Apply Parameters")
                if submitted:
                    try:
                        params = json.loads(params_text or "{}")
                        mutate_graph({"type": "set_params", "node_id": node_id, "params": params})
                    except json.JSONDecodeError as exc:
                        st.error(f"Invalid parameter object: {exc}")

        st.divider()
        insert_col, remove_col = st.columns([1, 1])
        with insert_col:
            st.write("Insert Before Model")
            if preprocessing_ops:
                root_node = next((node for node in graph.get("nodes", []) if node.get("id") == root_id), {})
                old_inputs = root_node.get("inputs", [])
                with st.form("insert_preprocessing"):
                    new_id = st.text_input("Node id", value=f"prep_{len(node_ids) + 1}")
                    new_op = st.selectbox("Preprocessing operation", preprocessing_ops, key="insert_op")
                    submitted = st.form_submit_button("Insert")
                    if submitted:
                        mutate_graph(
                            {
                                "type": "add",
                                "node": {"id": new_id, "operation": new_op, "params": {}, "inputs": old_inputs},
                                "rewire_input_of": root_id,
                            }
                        )
            else:
                st.caption("No safe preprocessing operations are exposed for this task.")

        with remove_col:
            st.write("Remove Node")
            removable = [node_id for node_id in node_ids if node_id != root_id]
            if removable:
                with st.form("remove_node"):
                    node_id = st.selectbox("Preprocessing node", removable, key="remove_node_id")
                    remove = st.form_submit_button("Remove Node")
                    if remove:
                        mutate_graph({"type": "remove", "node_id": node_id})
            else:
                st.caption("The graph currently has only the model node.")

    with operations_tab:
        render_operation_catalog(config, task_type, "editor")
        strategy_hints = config.get("training_strategy_hints", {}).get(task_type, []) or []
        if strategy_hints:
            st.write("Training strategy hints")
            st.dataframe(pd.DataFrame(strategy_hints), use_container_width=True, hide_index=True)

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
    if st.button("Evaluate Approved Graph", type="primary", use_container_width=True, key="editor_evaluate_approved_graph"):
        payload = current_payload(
            {
                "iterations": 1,
                "initial_graph": approved_graph,
            }
        )
        try:
            stream_run(payload)
            if st.session_state.get("result"):
                st.success("GraphAutoML run completed.")
        except Exception as exc:
            st.error(f"Run failed: {exc}")


def render_engineer_report(engineer: Dict[str, Any]) -> None:
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
            st.write("Reference label mapping")
            mapping_rows = [
                {"label": label, "code": code}
                for label, code in target_info["reference_encoding"].items()
            ]
            st.dataframe(pd.DataFrame(mapping_rows), use_container_width=True, hide_index=True)
        if target_info.get("sample_values"):
            st.caption("Target sample: " + ", ".join(target_info["sample_values"][:8]))

    notes = engineer.get("training_notes", []) or []
    if notes:
        st.write("Training notes")
        for note in notes:
            st.write(f"- {note}")

    if engineer.get("graph_error"):
        st.error(engineer["graph_error"])

    render_split_info(engineer.get("split_info", {}) or {})

    train_metrics = engineer.get("train_metrics", {}) or {}
    test_metrics = engineer.get("test_metrics", {}) or {}
    if train_metrics or test_metrics:
        train_col, test_col = st.columns(2)
        with train_col:
            render_metric_table("Train metrics", train_metrics)
        with test_col:
            render_metric_table("Test metrics (hold-out)", test_metrics)
    else:
        metrics = engineer.get("graph_metrics", {}) or {}
        if metrics:
            render_metric_table("Test metrics (hold-out)", metrics)

    tuned_nodes = engineer.get("tuned_nodes", []) or []
    if tuned_nodes:
        st.write("Tuned hyperparameters")
        rows = [
            {
                "node": node.get("id", ""),
                "operation": node.get("operation", ""),
                "params": format_params(node.get("tuned_params", {}) or {}),
            }
            for node in tuned_nodes
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    render_diagnostics(engineer.get("diagnostics", []), "Engineer diagnostics", use_expander=False)


def render_critic_feedback(critic: Dict[str, Any]) -> None:
    st.markdown("#### Critic Feedback")
    st.write(critic.get("assessment", "No critic assessment."))

    cols = st.columns(2)
    cols[0].metric("Decision", critic.get("winner", ""))
    cols[1].metric("Suggested changes", len(critic.get("suggested_mutations", []) or []))

    if critic.get("strengths"):
        st.write("What works")
        for item in critic["strengths"]:
            st.write(f"- {item}")
    if critic.get("weaknesses"):
        st.write("What should improve")
        for item in critic["weaknesses"]:
            st.write(f"- {item}")
    if critic.get("improvement_plan"):
        st.write("Improvement plan")
        for item in critic["improvement_plan"]:
            st.write(f"- {item}")
    if critic.get("suggested_mutations"):
        st.write(
            "Concrete mutations for Architect (alternatives - pick the ones you want applied; "
            "combining incompatible ones may produce an invalid graph)"
        )
        rows = mutation_rows(critic["suggested_mutations"])
        for idx, (mutation, row) in enumerate(zip(critic["suggested_mutations"], rows)):
            label = f"**{row['action']}** `{row['node']}`"
            if row.get("target"):
                label += f" -> `{row['target']}`"
            if row.get("details"):
                label += f" - {row['details']}"
            st.checkbox(label, value=(idx == 0), key=f"critic_mut_{idx}")

    render_diagnostics(critic.get("diagnostics", []), "Critic diagnostics", use_expander=False)


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
    if selected:
        st.success(
            f"Architect prepared a new draft from {len(selected)} selected mutation(s). "
            "Approve it to make it the active pipeline."
        )
    else:
        st.success("Architect prepared a new draft (no mutations selected; LLM-driven revision). Approve it to make it the active pipeline.")


def request_engineer_tuning(iteration: Dict[str, Any], iterations: int) -> Dict[str, Any]:
    graph = st.session_state.get("approved_graph") or iteration.get("architect", {}).get("graph", {})
    payload = current_payload({"graph": graph, "iterations": int(iterations)})
    result = post_json("/engineer/tune", payload, timeout=1800)
    st.session_state.tune_last_result = result

    if result.get("error"):
        return result

    archive_current_result("Before hyperparameter tuning")

    architect = iteration.setdefault("architect", {})
    tuned_graph = result.get("graph") or graph
    architect["graph"] = tuned_graph
    st.session_state.graph = tuned_graph
    st.session_state.approved_graph = tuned_graph
    st.session_state.graph_approved = True

    previous = iteration.get("engineer", {}) or {}
    metrics = result.get("metrics", {}) or {}
    iteration["engineer"] = {
        **previous,
        "graph_score": result.get("score", metrics.get("primary_score", previous.get("graph_score", 0))),
        "graph_metrics": metrics,
        "train_metrics": result.get("train_metrics", {}) or {},
        "test_metrics": result.get("test_metrics", {}) or metrics,
        "split_info": result.get("split_info", {}) or {},
        "tuned_nodes": result.get("tuned_nodes", []) or [],
        "graph_error": result.get("error", "") or "",
        "target_info": result.get("target_info", {}) or previous.get("target_info", {}),
        "training_notes": result.get("training_notes", []) or previous.get("training_notes", []),
        "diagnostics": result.get("diagnostics", []) or [],
        "tool_calls": previous.get("tool_calls", []),
    }
    return result


def render_engineer_tuning_controls(iteration: Dict[str, Any]) -> None:
    st.markdown("#### Engineer Tuning")
    st.caption(
        "Runs the existing MCP `tune_graph_hyperparameters` tool on the approved graph, "
        "then recalculates train and hold-out test metrics with the selected split."
    )
    tune_cols = st.columns([1, 2])
    iterations = tune_cols[0].number_input(
        "Tuning iterations",
        min_value=1,
        max_value=100,
        value=int(st.session_state.get("tune_iterations", 30)),
        step=5,
        key="tune_iterations",
    )
    if tune_cols[1].button("Tune Approved Graph", use_container_width=True, key="engineer_tune_approved"):
        try:
            with st.spinner("Engineer is tuning the approved graph..."):
                result = request_engineer_tuning(iteration, int(iterations))
            if result.get("error"):
                st.warning(result.get("error"))
                render_diagnostics(result.get("diagnostics", []) or [], "Tuning diagnostics", use_expander=False)
            else:
                st.session_state.tune_success_message = (
                    "Tuning completed. Metrics were recalculated on train and test splits."
                )
                st.rerun()
        except Exception as exc:
            st.error(f"Engineer tuning failed: {exc}")


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
        "critic": critic.get("winner", ""),
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


def set_m4_graph(result: Dict[str, Any]) -> None:
    st.session_state.m4_graph = result.get("graph", {}) or {}
    st.session_state.m4_profile = result.get("profile", {}) or {}
    st.session_state.m4_architect_analysis = result.get("analysis", "")
    st.session_state.m4_architect_reasoning = result.get("reasoning", "")
    st.session_state.m4_architect_diagnostics = result.get("diagnostics", []) or []
    st.session_state.m4_architect_tool_calls = result.get("tool_calls", []) or []
    st.session_state.m4_graph_approved = False


def approve_m4_graph() -> None:
    graph = st.session_state.get("m4_graph") or {}
    if graph:
        st.session_state.m4_approved_graph = graph
        st.session_state.m4_graph_approved = True


def render_m4_benchmark_result(result: Dict[str, Any]) -> None:
    if not result:
        return

    if result.get("error"):
        st.error(result["error"])
        render_diagnostics(result.get("diagnostics", []) or [], "Benchmark diagnostics", use_expander=False)
        return

    st.markdown("#### M4 Benchmark Result")
    metrics = result.get("test_metrics", {}) or result.get("metrics", {}) or {}
    train_metrics = result.get("train_metrics", {}) or {}
    primary_metric = metrics.get("primary_metric") or "score"
    primary_value = metrics.get("primary_metric_value", result.get("score", 0))
    train_value = train_metrics.get("primary_metric_value", "")
    dataset = result.get("dataset", {}) or {}

    cols = st.columns(4)
    cols[0].metric(f"Test {primary_metric}", format_number(primary_value))
    cols[1].metric(f"Train {primary_metric}", format_number(train_value))
    cols[2].metric("Groups", len(dataset.get("groups", []) or []))
    cols[3].metric("Window", dataset.get("window_length", ""))

    render_graph(result.get("graph", {}), show_details=False)
    render_split_info(result.get("split_info", {}) or {})

    train_col, test_col = st.columns(2)
    with train_col:
        render_metric_table("Train metrics", train_metrics)
    with test_col:
        render_metric_table("Test metrics (hold-out)", metrics)

    if result.get("class_report"):
        st.write("Per-group test metrics")
        st.dataframe(pd.DataFrame(result["class_report"]), use_container_width=True, hide_index=True)

    if result.get("group_counts"):
        st.write("Group split")
        st.dataframe(pd.DataFrame(result["group_counts"]), use_container_width=True, hide_index=True)

    source_files = dataset.get("source_files", []) or []
    if source_files:
        with st.expander("M4 source files"):
            st.dataframe(pd.DataFrame(source_files), use_container_width=True, hide_index=True)


def benchmarks_tab(config: Dict[str, Any]) -> None:
    st.subheader("Benchmarks")
    bench_config = (config.get("benchmarks", {}) or {}).get("m4_classification", {})
    groups = bench_config.get("groups") or DEFAULT_M4_GROUPS
    metric_options = config.get("metrics_by_task", {}).get("ts_classification") or DEFAULT_METRICS_BY_TASK["ts_classification"]

    st.write("M4 frequency-group classification")
    st.caption(
        "Uses real M4 train CSV files downloaded through datasetsforecast. "
        "Each time series is one sample; the target class is its frequency group."
    )

    settings_col, graph_col = st.columns([1, 1])
    with settings_col:
        selected_groups = st.multiselect(
            "M4 groups",
            groups,
            default=groups,
            help="At least two groups are recommended because the benchmark is classification by frequency group.",
        )
        n_per_group = st.number_input(
            "Series per group",
            min_value=10,
            max_value=1000,
            value=int(bench_config.get("default_n_per_group", 100)),
            step=10,
        )
        window_length = st.number_input(
            "Window length",
            min_value=8,
            max_value=500,
            value=int(bench_config.get("default_window_length", 50)),
            step=5,
        )
        test_size = st.slider(
            "M4 test size",
            min_value=0.05,
            max_value=0.5,
            value=float(st.session_state.get("m4_test_size", 0.3)),
            step=0.05,
            key="m4_test_size",
        )
        metric = st.selectbox(
            "M4 primary metric",
            metric_options,
            index=metric_options.index("f1") if "f1" in metric_options else 0,
        )
        standardize = st.checkbox("Standardize each series", value=True)
        message = st.text_area(
            "Request to Architect",
            placeholder="Example: prefer fast feature extraction, avoid deep neural models, or try frequency-domain features.",
            height=90,
            key="m4_architect_message",
        )
        if st.button("Ask Architect For M4 Graph", type="primary", use_container_width=True, key="m4_ask_architect"):
            payload = {
                "message": message,
                "current_graph": st.session_state.get("m4_graph"),
                "groups": selected_groups or groups,
                "n_per_group": int(n_per_group),
                "window_length": int(window_length),
                "test_size": float(test_size),
                "primary_metric": metric,
                "standardize": bool(standardize),
            }
            try:
                with st.spinner("Architect is drafting an M4 benchmark graph..."):
                    result = post_json("/benchmarks/m4/architect", payload, timeout=240)
                set_m4_graph(result)
                st.success("Architect proposed an M4 benchmark graph. Approve it before running the benchmark.")
            except Exception as exc:
                st.error(f"M4 Architect failed: {exc}")

    graph = st.session_state.get("m4_graph", {}) or {}
    with graph_col:
        status = "approved" if st.session_state.get("m4_graph_approved") else "draft"
        st.write(f"M4 Architect graph ({status})")
        if graph:
            render_graph(graph, show_details=False)
            st.dataframe(pd.DataFrame(graph_rows(graph)), use_container_width=True, hide_index=True)
            action_cols = st.columns(2)
            if action_cols[0].button("Approve M4 Graph", use_container_width=True, key="m4_approve_graph"):
                approve_m4_graph()
                st.success("M4 graph approved for benchmark.")
            if action_cols[1].button("Discard M4 Draft", use_container_width=True, key="m4_discard_graph"):
                st.session_state.m4_graph = st.session_state.get("m4_approved_graph", {}) or {}
                st.session_state.m4_graph_approved = bool(st.session_state.get("m4_graph"))
                st.info("M4 draft discarded.")
            if st.session_state.get("m4_architect_analysis"):
                st.write("Analysis")
                st.write(st.session_state.m4_architect_analysis)
            if st.session_state.get("m4_architect_reasoning"):
                st.write("Reasoning")
                st.write(st.session_state.m4_architect_reasoning)
            render_diagnostics(st.session_state.get("m4_architect_diagnostics", []), "M4 Architect diagnostics")
            render_tool_calls(st.session_state.get("m4_architect_tool_calls", []), "M4 Architect tool calls")
        else:
            st.info("Ask Architect to draft an M4 graph first.")
            st.write("Available ts_classification operations")
            render_operation_catalog(config, "ts_classification", "m4_benchmark")

    if len(selected_groups or []) < 2:
        st.warning("Select at least two M4 groups for a classification benchmark.")

    approved_graph = st.session_state.get("m4_approved_graph") or {}
    can_run = bool(approved_graph) and st.session_state.get("m4_graph_approved") and len(selected_groups or []) >= 2
    if st.button(
        "Run M4 Benchmark",
        type="primary",
        use_container_width=True,
        key="run_m4_benchmark",
        disabled=not can_run,
    ):
        payload = {
            "graph": approved_graph,
            "groups": selected_groups or groups,
            "n_per_group": int(n_per_group),
            "window_length": int(window_length),
            "test_size": float(test_size),
            "primary_metric": metric,
            "standardize": bool(standardize),
        }
        try:
            with st.spinner("Downloading/loading M4 and running the benchmark..."):
                st.session_state.m4_benchmark_result = post_json("/benchmarks/m4", payload, timeout=3600)
            st.success("M4 benchmark completed.")
        except Exception as exc:
            st.error(f"M4 benchmark failed: {exc}")

    if not approved_graph:
        st.caption("Benchmark run is disabled until an Architect graph is approved.")

    render_m4_benchmark_result(st.session_state.get("m4_benchmark_result", {}) or {})


def results_tab() -> None:
    st.subheader("Evaluation Result")
    result = st.session_state.get("result")
    if not result:
        st.info("Evaluate an approved graph to see Engineer and Critic feedback.")
        return

    item = latest_iteration(result)
    if not item:
        st.error("No successful evaluation found.")
        return

    tune_message = st.session_state.get("tune_success_message", "")
    if tune_message:
        del st.session_state["tune_success_message"]
    if tune_message:
        st.success(tune_message)

    engineer = item.get("engineer", {})
    critic = item.get("critic", {})

    metrics = engineer.get("test_metrics", {}) or engineer.get("graph_metrics", {}) or {}
    primary_metric = metrics.get("primary_metric") or result.get("primary_metric") or "score"
    primary_value = metrics.get("primary_metric_value", engineer.get("graph_score", 0))

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Test {primary_metric}", format_number(primary_value))
    col2.metric("Critic decision", critic.get("winner", ""))
    col3.metric("Suggested changes", len(critic.get("suggested_mutations", []) or []))

    st.markdown("#### Evaluated Graph")
    render_graph(item.get("architect", {}).get("graph", {}), show_details=False)

    render_engineer_tuning_controls(item)
    engineer = item.get("engineer", {})
    render_engineer_report(engineer)
    render_evaluation_history(result)
    if engineer.get("tuned_nodes"):
        st.info("Critic feedback below is from the last full evaluation. Run Evaluate Approved Graph again to reassess the tuned parameters.")
    render_critic_feedback(critic)

    st.markdown("#### Next Pipeline Decision")
    msg = st.text_area(
        "Optional note for Architect",
        placeholder="Example: prefer a simpler model, avoid heavy operations, or apply only the model replacement.",
        height=90,
    )
    col_apply, col_keep = st.columns(2)
    if col_apply.button("Ask Architect To Draft Critic Changes", type="primary", use_container_width=True, key="critic_draft_changes"):
        try:
            request_architect_revision(item, msg)
        except Exception as exc:
            st.error(f"Architect revision failed: {exc}")
    if col_keep.button("Keep Current Approved Pipeline", use_container_width=True, key="critic_keep_current"):
        discard_draft()
        st.info("Current approved pipeline kept. No new draft accepted.")


def report_tab() -> None:
    st.subheader("Report")
    result = st.session_state.get("result")
    if not result:
        st.info("No report yet.")
        return

    report = result.get("report", {})
    st.markdown(f"### {report.get('title', 'GraphAutoML report')}")
    st.write(report.get("summary", ""))
    if report.get("methodology"):
        st.write("Methodology")
        st.write(report["methodology"])
    if report.get("results"):
        st.write("Results")
        st.write(report["results"])
    if report.get("recommendations"):
        st.write("Recommendations")
        for item in report["recommendations"]:
            st.write(f"- {item}")


def tools_tab(tools: Dict[str, Any]) -> None:
    st.subheader("MCP Tools")
    if not tools:
        st.info("Tool metadata is unavailable.")
        return
    for agent, agent_tools in tools.items():
        st.write(agent.title())
        st.dataframe(pd.DataFrame(agent_tools), use_container_width=True, hide_index=True)


def main() -> None:
    st.title("GraphAutoML")
    st.caption("LLM agents compose, tune, validate and report Fedot.Industrial pipeline graphs through MCP tools.")
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
