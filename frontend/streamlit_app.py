import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="GraphAutoML", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")
TASK_TYPES = ["classification", "regression", "ts_classification", "ts_regression", "ts_forecasting"]


def shared_data_dir() -> Path:
    docker_dir = Path("/app/data")
    if docker_dir.exists():
        return docker_dir
    repo_data = Path(__file__).resolve().parents[1] / "data"
    repo_data.mkdir(exist_ok=True)
    return repo_data


@st.cache_data(ttl=60)
def load_config() -> Dict[str, Any]:
    try:
        response = requests.get(f"{BACKEND_URL}/config", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {"operations": {}, "metrics_by_task": {}, "supported_tasks": TASK_TYPES}


@st.cache_data(ttl=60)
def load_tools() -> Dict[str, Any]:
    try:
        response = requests.get(f"{BACKEND_URL}/tools", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


def post_json(path: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    response = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
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
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    data_dir = shared_data_dir()
    filename = f"uploaded_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = data_dir / filename
    df.to_csv(path, index=False)
    st.session_state.csv_path = str(path)
    return str(path)


def current_payload(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = {
        "csv_path": st.session_state.get("csv_path", ""),
        "target_column": st.session_state.get("target_column", ""),
        "task_type": st.session_state.get("task_type", "classification"),
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


def approve_graph() -> None:
    if st.session_state.get("graph"):
        st.session_state.graph_approved = True


def root_node_id(graph: Dict[str, Any]) -> str:
    nodes = graph.get("nodes", [])
    ids = {node.get("id") for node in nodes}
    children = {inp for node in nodes for inp in node.get("inputs", [])}
    roots = [node_id for node_id in ids - children if node_id]
    return roots[0] if roots else ""


def require_dataset() -> bool:
    if not st.session_state.get("csv_path") or "df" not in st.session_state:
        st.info("Upload a CSV file to start.")
        return False
    return True


def render_graph(graph: Dict[str, Any]) -> None:
    if not graph:
        st.info("No graph proposed yet.")
        return
    st.graphviz_chart(graph_to_dot(graph), use_container_width=True)
    with st.expander("Graph JSON"):
        st.json(graph)
    if st.session_state.get("mermaid"):
        with st.expander("Mermaid"):
            st.code(st.session_state.mermaid, language="mermaid")


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


def render_diagnostics(diagnostics: List[Dict[str, Any]], title: str = "Diagnostics") -> None:
    if not diagnostics:
        return
    with st.expander(title, expanded=True):
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
                with st.expander("Technical message"):
                    st.code(str(item["technical_message"])[:4000])


def operation_rows(config: Dict[str, Any], task_type: str) -> List[Dict[str, str]]:
    catalog = config.get("operation_catalog", {}).get(task_type, {})
    if catalog:
        rows = []
        for group, items in catalog.items():
            for item in items:
                rows.append({
                    "group": group,
                    "operation": item.get("operation", ""),
                    "description": item.get("description", ""),
                })
        return rows

    operations = config.get("operations", {}).get(task_type, {})
    descriptions = config.get("operation_descriptions", {})
    return [
        {"group": group, "operation": name, "description": descriptions.get(name, "")}
        for group, names in operations.items()
        for name in names
    ]


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
            "params": json.dumps(node.get("params", {}), ensure_ascii=False),
        })
    return rows


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
    total_steps = max(1, payload.get("iterations", 1) * 3 + 1)
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
                lines.append(event.get("message", ""))
            elif event_type == "agent_start":
                lines.append(f"Iteration {event.get('iteration')}: {event.get('agent')} started")
            elif event_type == "agent_done":
                lines.append(f"{event.get('agent')}: {event.get('summary', '')}")
                if event.get("diagnostics"):
                    for diagnostic in event["diagnostics"]:
                        lines.append(f"{event.get('agent')} diagnostic: {diagnostic.get('summary', '')}")
                done_steps += 1
                progress.progress(min(done_steps / total_steps, 1.0))
            elif event_type == "diagnostics":
                for diagnostic in event.get("diagnostics", []):
                    lines.append(f"{event.get('agent')} diagnostic: {diagnostic.get('summary', '')}")
            elif event_type == "iteration_done":
                lines.append(
                    f"Iteration {event.get('iteration')} done: graph={event.get('graph_score', 0):.4f}, "
                    f"baseline={event.get('best_baseline_score', 0):.4f}, winner={event.get('winner')}"
                )
                if event.get("graph"):
                    st.session_state.graph = event["graph"]
                    st.session_state.mermaid = event.get("mermaid", "")
            elif event_type == "error":
                lines.append(f"ERROR: {event.get('message')}")
            elif event_type == "complete":
                st.session_state.result = event.get("result", {})
                progress.progress(1.0)
                lines.append("Run completed.")

            log_box.markdown("\n\n".join(f"- {line}" for line in lines[-14:]))


def sidebar() -> None:
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
            if st.session_state.task_type == "ts_forecasting":
                st.session_state.forecast_length = st.number_input("Forecast length", min_value=1, value=14)
            else:
                st.session_state.forecast_length = None
            st.session_state.iterations = st.slider("Iterations", min_value=1, max_value=5, value=2)

        st.divider()
        st.caption(f"Backend: {BACKEND_URL}")
        try:
            health = requests.get(f"{BACKEND_URL}/health", timeout=2).json()
            st.success(health.get("status", "connected"))
        except Exception:
            st.error("backend unavailable")


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
        rows = operation_rows(config, task_type)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        message = st.text_area(
            "Request to Architect",
            placeholder="Example: prefer a simpler linear model, use frequency features for a signal task, or keep the graph conservative.",
            height=110,
        )
        ask_col, default_col = st.columns(2)
        if ask_col.button("Ask Architect", type="primary", use_container_width=True):
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

        if default_col.button("Use Reliable Default", use_container_width=True):
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
            if st.button("Approve Draft", use_container_width=True):
                approve_graph()
                st.success("Graph approved.")
        render_tool_calls(st.session_state.get("architect_tool_calls", []), "Architect tool calls")


def graph_editor_tab(config: Dict[str, Any]) -> None:
    st.subheader("Graph Editor")
    graph = st.session_state.get("graph", {})
    if not graph:
        st.info("Ask Architect for a graph, then edit it here.")
        return

    task_type = graph.get("task_type", st.session_state.get("task_type", "classification"))
    operations = config.get("operations", {}).get(task_type, {})
    preprocessing_ops = operations.get("preprocessing", [])
    model_ops = operations.get("models", [])
    node_ids = [node.get("id") for node in graph.get("nodes", [])]
    root_id = root_node_id(graph)

    status_col, approve_col = st.columns([2, 1])
    status_col.caption("Approved" if st.session_state.get("graph_approved") else "Draft")
    if approve_col.button("Approve Edited Graph", use_container_width=True):
        approve_graph()
        st.success("Graph approved.")

    st.dataframe(pd.DataFrame(graph_rows(graph)), use_container_width=True, hide_index=True)
    render_graph(graph)

    catalog_rows = operation_rows(config, task_type)
    if catalog_rows:
        with st.expander("Operation catalog"):
            st.dataframe(pd.DataFrame(catalog_rows), use_container_width=True, hide_index=True)

    col_insert, col_model, col_params = st.columns(3)

    with col_insert:
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

    with col_model:
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

        removable = [node_id for node_id in node_ids if node_id != root_id]
        if removable:
            with st.form("remove_node"):
                node_id = st.selectbox("Preprocessing node", removable, key="remove_node_id")
                remove = st.form_submit_button("Remove Node")
                if remove:
                    mutate_graph({"type": "remove", "node_id": node_id})

    with col_params:
        st.write("Set Parameters")
        with st.form("params_node"):
            node_id = st.selectbox("Node", node_ids, key="params_node_id")
            current = next((node for node in graph.get("nodes", []) if node.get("id") == node_id), {})
            params_text = st.text_area(
                "Params JSON",
                value=json.dumps(current.get("params", {}) or {}, ensure_ascii=False, indent=2),
                height=130,
            )
            submitted = st.form_submit_button("Apply params")
            if submitted:
                try:
                    params = json.loads(params_text or "{}")
                    mutate_graph({"type": "set_params", "node_id": node_id, "params": params})
                except json.JSONDecodeError as exc:
                    st.error(f"Invalid JSON: {exc}")


def run_tab() -> None:
    st.subheader("Run Approved Graph")
    if not require_dataset():
        return
    graph = st.session_state.get("graph")
    if not graph:
        st.info("Approve a graph first in the Architect tab.")
        return

    if not st.session_state.get("graph_approved"):
        st.warning("Graph is still a draft. Running will approve this exact version.")
    render_graph(graph)
    if st.button("Approve Graph and Run", type="primary", use_container_width=True):
        approve_graph()
        payload = current_payload(
            {
                "iterations": int(st.session_state.get("iterations", 2)),
                "initial_graph": graph,
            }
        )
        try:
            stream_run(payload)
            if st.session_state.get("result"):
                st.success("GraphAutoML run completed.")
        except Exception as exc:
            st.error(f"Run failed: {exc}")


def results_tab() -> None:
    st.subheader("Results")
    result = st.session_state.get("result")
    if not result:
        st.info("Run an approved graph to see results.")
        return

    summary = result.get("summary", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Iterations", summary.get("successful_iterations", 0))
    col2.metric("Max graph", f"{summary.get('max_graph_score', 0):.4f}")
    col3.metric("Max baseline", f"{summary.get('max_baseline_score', 0):.4f}")
    col4.metric("Graph wins", summary.get("graph_wins_count", 0))

    rows = []
    for item in result.get("iterations", []):
        if "error" in item:
            rows.append({"iteration": item.get("iteration"), "status": "failed", "error": item.get("error")})
            continue
        engineer = item.get("engineer", {})
        critic = item.get("critic", {})
        rows.append(
            {
                "iteration": item.get("iteration"),
                "graph_score": engineer.get("graph_score", 0),
                "best_baseline": engineer.get("best_baseline_score", 0),
                "baseline_name": engineer.get("best_baseline_name", ""),
                "winner": critic.get("winner", ""),
                "stop": critic.get("should_stop", False),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for item in result.get("iterations", []):
        if "error" in item:
            continue
        with st.expander(f"Iteration {item.get('iteration')} details"):
            st.write("Graph")
            render_graph(item.get("architect", {}).get("graph", {}))
            st.write("Engineer metrics")
            st.json(item.get("engineer", {}).get("graph_metrics", {}))
            render_diagnostics(item.get("engineer", {}).get("diagnostics", []), "Engineer diagnostics")
            st.write("Critic")
            critic = item.get("critic", {})
            st.write(critic.get("assessment", ""))
            if critic.get("strengths"):
                st.write("Strengths")
                st.write(critic["strengths"])
            if critic.get("weaknesses"):
                st.write("Weaknesses")
                st.write(critic["weaknesses"])
            if critic.get("suggested_mutations"):
                st.write("Suggested mutations")
                st.json(critic["suggested_mutations"])
            render_diagnostics(critic.get("diagnostics", []), "Critic diagnostics")


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
    if report.get("best_graph_mermaid"):
        with st.expander("Best graph Mermaid"):
            st.code(report["best_graph_mermaid"], language="mermaid")


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
    sidebar()
    config = load_config()
    tools = load_tools()

    tabs = st.tabs(["Data", "Architect", "Graph Editor", "Run", "Results", "Report", "MCP Tools"])
    with tabs[0]:
        data_tab()
    with tabs[1]:
        architect_tab(config)
    with tabs[2]:
        graph_editor_tab(config)
    with tabs[3]:
        run_tab()
    with tabs[4]:
        results_tab()
    with tabs[5]:
        report_tab()
    with tabs[6]:
        tools_tab(tools)


if __name__ == "__main__":
    main()
