import streamlit as st
import requests
import pandas as pd
import json
import os

st.set_page_config(page_title="Multi-Agent ML System", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8001")

st.title("Multi-Agent ML System with Fedot.Industrial")

tab_data, tab_architect, tab_engineer, tab_critic, tab_report, tab_summary = st.tabs([
    "Data", "Architect", "Engineer", "Critic", "Report", "Summary"
])

# ============== SIDEBAR ==============

with st.sidebar:
    st.header("Configuration")

    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.write(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    if "df" in st.session_state:
        df = st.session_state.df

        target_column = st.selectbox(
            "Target Column",
            options=df.columns,
            key="target_select",
        )

        task_type = st.selectbox(
            "Task Type",
            options=[
                "classification", "regression",
                "ts_forecasting", "ts_classification", "ts_regression",
                "anomaly_detection",
            ],
            key="task_select",
        )

        # Forecast length for ts_forecasting
        forecast_length = None
        if task_type == "ts_forecasting":
            forecast_length = st.number_input(
                "Forecast Length (steps)", min_value=1, max_value=1000, value=14,
                key="forecast_length",
            )

        iterations = st.slider(
            "Iterations", min_value=1, max_value=5, value=3, key="iterations_slider"
        )

        # Fedot.Industrial Configuration
        with st.expander("Fedot.Industrial Config", expanded=False):
            fedot_preset = st.selectbox(
                "Preset",
                options=["auto", "best_quality", "fast_train", "stable", "light", "ultra-light"],
                key="fedot_preset",
            )
            fedot_timeout = st.slider(
                "Timeout (min)", min_value=1, max_value=30, value=5, key="fedot_timeout"
            )

            # Metrics depend on task type
            if task_type in ("classification", "ts_classification", "anomaly_detection"):
                metric_options = ["auto", "f1", "roc_auc", "accuracy", "precision", "logloss"]
            elif task_type == "ts_forecasting":
                metric_options = ["auto", "rmse", "mse", "mae", "mape", "smape", "r2"]
            else:
                metric_options = ["auto", "r2", "rmse", "mse", "mae"]

            fedot_metric = st.selectbox("Metric", options=metric_options, key="fedot_metric")
            fedot_njobs = st.slider("N Jobs", min_value=-1, max_value=8, value=2, key="fedot_njobs")
            fedot_cv = st.number_input("CV Folds (0 = auto)", min_value=0, max_value=10, value=0, key="fedot_cv")

            # Strategy (task-dependent)
            strategy_options = ["none"]
            if task_type == "ts_forecasting":
                strategy_options += ["forecasting_assumptions", "forecasting_exogenous", "federated_automl", "kernel_automl"]
            elif task_type in ("classification", "regression", "ts_classification", "ts_regression"):
                strategy_options += ["federated_automl", "kernel_automl", "lora_strategy", "sampling_strategy"]
            fedot_strategy = st.selectbox("Strategy", options=strategy_options, key="fedot_strategy")

        run_button = st.button("Run Orchestration", key="run_orchestration")

        if run_button:
            # Input validation
            validation_ok = True

            if df.empty:
                st.error("Dataset is empty. Upload a valid CSV.")
                validation_ok = False

            if target_column not in df.columns:
                st.error(f"Target column '{target_column}' not found in data.")
                validation_ok = False
            elif df[target_column].isnull().all():
                st.error(f"Target column '{target_column}' has no values.")
                validation_ok = False

            if task_type == "ts_forecasting" and (not forecast_length or forecast_length < 1):
                st.error("Forecast length must be >= 1 for ts_forecasting.")
                validation_ok = False

            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) == 0 and task_type in ("regression", "ts_regression", "ts_forecasting"):
                st.error("No numeric columns found. Regression/forecasting requires numeric features.")
                validation_ok = False

            if not validation_ok:
                st.stop()

            try:
                unique_name = f"uploaded_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                shared_path = os.path.join("/app/data", unique_name)
                df.to_csv(shared_path, index=False)

                # Build initial Fedot config
                initial_fedot_config = {
                    "preset": fedot_preset,
                    "timeout": fedot_timeout,
                    "n_jobs": fedot_njobs,
                }
                if fedot_metric != "auto":
                    initial_fedot_config["metric"] = fedot_metric
                if fedot_cv > 0:
                    initial_fedot_config["cv_folds"] = fedot_cv
                if fedot_strategy != "none":
                    initial_fedot_config["strategy"] = fedot_strategy

                payload = {
                    "csv_path": shared_path,
                    "target_column": target_column,
                    "task_type": task_type,
                    "iterations": iterations,
                    "fedot_url": "http://fedot-server:8000",
                    "initial_fedot_config": initial_fedot_config,
                }
                if task_type == "ts_forecasting" and forecast_length:
                    payload["forecast_length"] = forecast_length

                # SSE streaming progress
                progress_container = st.container()
                progress_log = progress_container.empty()
                progress_bar = progress_container.progress(0)
                log_lines = []

                try:
                    with requests.post(
                        f"{BACKEND_URL}/orchestrate/stream",
                        json=payload, stream=True, timeout=900,
                    ) as resp:
                        resp.raise_for_status()
                        total_steps = iterations * 3 + 1  # 3 agents per iter + scribe
                        step = 0

                        for line in resp.iter_lines(decode_unicode=True):
                            if not line or not line.startswith("data: "):
                                continue
                            try:
                                evt = json.loads(line[6:])
                            except json.JSONDecodeError:
                                continue

                            etype = evt.get("event", "")

                            if etype == "status":
                                log_lines.append(f"**{evt['message']}**")
                            elif etype == "agent_start":
                                agent = evt.get("agent", "")
                                it = evt.get("iteration", "")
                                step_label = evt.get("step", "")
                                log_lines.append(f"  `[Iter {it}, {step_label}]` Starting **{agent}**...")
                            elif etype == "agent_done":
                                agent = evt.get("agent", "")
                                summary = evt.get("summary", "")
                                tc = evt.get("tool_calls_count", 0)
                                log_lines.append(f"  `[{agent}]` Done ({tc} tool calls): {summary}")
                                step += 1
                                progress_bar.progress(min(step / total_steps, 1.0))
                            elif etype == "iteration_done":
                                it = evt.get("iteration", "?")
                                score = evt.get("best_score", 0)
                                model = evt.get("best_model", "N/A")
                                winner = evt.get("winner", "N/A")
                                log_lines.append(f"**Iteration {it} done:** best={model} ({score:.4f}), winner={winner}")
                                log_lines.append("---")
                            elif etype == "error":
                                log_lines.append(f"**ERROR:** {evt.get('message', 'Unknown')}")
                            elif etype == "complete":
                                result = evt.get("result", {})
                                st.session_state.result = result
                                progress_bar.progress(1.0)
                                log_lines.append("**Orchestration completed!**")

                            # Update live log
                            progress_log.markdown("\n\n".join(log_lines[-15:]))

                    if "result" in st.session_state:
                        st.success("Orchestration completed! See tabs for detailed results.")
                    else:
                        st.warning("Stream ended without final result. Check logs.")

                except requests.Timeout:
                    st.error("Request timeout (15 min)")
                except requests.ConnectionError:
                    st.error("Connection lost to backend")

            except Exception as e:
                st.error(f"Error: {e}")


# ============== LOAD MCP TOOLS INFO ==============

@st.cache_data(ttl=60)
def load_mcp_tools():
    """Fetch MCP tool descriptions from backend /tools endpoint."""
    try:
        resp = requests.get(f"{BACKEND_URL}/tools", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    # Fallback
    return {
        "architect": [
            {"name": "get_data_profile", "description": "Profile dataset: statistics, issues, recommendations"},
            {"name": "get_available_operations", "description": "Get Fedot.Industrial models/preprocessing/strategies"},
            {"name": "propose_baselines", "description": "Select sklearn baselines (classification/regression)"},
            {"name": "propose_fedot_config", "description": "Configure Fedot.Industrial (preset, strategy, operations...)"},
            {"name": "mutate_fedot_config", "description": "Adjust config based on Critic feedback"},
        ],
        "engineer": [
            {"name": "train_baseline", "description": "Train sklearn pipeline and get metrics"},
            {"name": "call_fedot", "description": "Call Fedot.Industrial with full parameter control"},
            {"name": "compare_results", "description": "Compare all models"},
        ],
        "critic": [
            {"name": "analyze_errors", "description": "Analyze prediction errors"},
            {"name": "get_feature_importance", "description": "Feature importance from tree-based models"},
            {"name": "explain_model", "description": "Explain Fedot model (point/recurrence/shap/lime)"},
            {"name": "get_additional_metrics", "description": "Calculate additional metrics"},
        ],
        "scribe": [
            {"name": "generate_report", "description": "Compile iteration results into report"},
        ],
    }


AGENT_INFO = {
    "architect": {
        "icon": "Architect Agent",
        "role": "Analyzes data, selects strategy and model configuration",
        "tasks": "classification, regression, ts_forecasting, ts_classification, ts_regression, anomaly_detection",
    },
    "engineer": {
        "icon": "Engineer Agent",
        "role": "Trains models, calls Fedot.Industrial, compares results",
        "tasks": "Trains sklearn baselines (tabular) + Fedot.Industrial (all task types)",
    },
    "critic": {
        "icon": "Critic Agent",
        "role": "Evaluates models, explains predictions, provides structured feedback",
        "tasks": "Error analysis, feature importance, model explanation, parameter suggestions",
    },
    "scribe": {
        "icon": "Scribe Agent",
        "role": "Generates comprehensive final report (called once)",
        "tasks": "Executive summary, methodology, results, recommendations",
    },
}


def render_agent_card(agent_key: str):
    """Render an agent capability card with MCP tools."""
    tools = load_mcp_tools()
    info = AGENT_INFO.get(agent_key, {})
    agent_tools = tools.get(agent_key, [])

    st.markdown(f"### {info.get('icon', agent_key)}")
    st.markdown(f"**Role:** {info.get('role', '')}")

    st.markdown("**MCP Tools:**")
    for tool in agent_tools:
        st.markdown(f"- `{tool['name']}` — {tool['description']}")

    st.markdown(f"**Scope:** {info.get('tasks', '')}")
    st.caption("Protocol: MCP (Model Context Protocol) via stdio transport")


# ============== HELPER FUNCTIONS ==============

def render_tool_calls(tool_calls, title="Tool Calls (MCP)"):
    """Render tool call log as expandable section."""
    if not tool_calls:
        return
    with st.expander(f"{title} ({len(tool_calls)} calls)", expanded=False):
        for i, tc in enumerate(tool_calls):
            st.markdown(f"**{i+1}. `{tc.get('tool', 'unknown')}`**")
            if tc.get("args"):
                st.json(tc["args"])
            result_str = tc.get("result", "")
            if len(str(result_str)) > 200:
                st.code(str(result_str)[:200] + "...", language="json")
            else:
                st.code(str(result_str), language="json")
            st.divider()


def render_fedot_config(config, title="Fedot.Industrial Config"):
    """Render Fedot config as a formatted card."""
    if not config:
        return
    st.markdown(f"**{title}:**")
    cols = st.columns(4)
    keys = list(config.keys())
    for i, key in enumerate(keys):
        with cols[i % 4]:
            st.metric(key, str(config[key]))


# ============== DATA TAB ==============

with tab_data:
    st.subheader("Data Preview and Statistics")

    if "df" in st.session_state:
        df = st.session_state.df

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())

        st.write("First 10 rows:")
        st.dataframe(df.head(10), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.write("Data Types:")
            st.dataframe(df.dtypes, use_container_width=True)
        with col_b:
            st.write("Statistics:")
            st.dataframe(df.describe(), use_container_width=True)

        # Data profile from orchestration
        if "result" in st.session_state:
            profile = st.session_state.result.get("profile", {})
            if profile:
                st.subheader("Data Profile (from DataProfiler)")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Issues:**")
                    for issue in profile.get("issues", []):
                        st.warning(issue)
                with col2:
                    st.markdown("**Recommendations:**")
                    for rec in profile.get("recommendations", []):
                        st.info(rec)
    else:
        st.info("Upload a CSV file to see data preview")


# ============== ARCHITECT TAB ==============

with tab_architect:
    st.subheader("Architect: Pipeline Selection & Fedot Config")

    if "result" in st.session_state:
        result = st.session_state.result
        iterations_data = result.get("iterations", [])

        for iter_num, iter_data in enumerate(iterations_data, 1):
            if "error" in iter_data:
                continue

            architect_info = iter_data.get("architect", {})
            st.markdown(f"### Iteration {iter_num}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Selected Baselines:**")
                for pipeline in architect_info.get("selected_baselines", architect_info.get("selected_pipelines", [])):
                    st.write(f"- {pipeline}")

                st.markdown("**Analysis:**")
                st.info(architect_info.get("analysis", "No analysis"))

            with col2:
                st.markdown("**Reasoning:**")
                st.write(architect_info.get("reasoning", "No reasoning"))

                # Show Fedot config
                fedot_cfg = architect_info.get("fedot_config", {})
                if fedot_cfg:
                    render_fedot_config(fedot_cfg, "Proposed Fedot Config")

            # Tool calls
            render_tool_calls(architect_info.get("tool_calls", []), "Architect Tool Calls")

            st.divider()

        if not any("error" not in d for d in iterations_data):
            st.warning("No successful architect analyses")
    else:
        render_agent_card("architect")


# ============== ENGINEER TAB ==============

with tab_engineer:
    st.subheader("Engineer: Training Results")

    if "result" in st.session_state:
        result = st.session_state.result
        iterations_data = result.get("iterations", [])

        for iter_num, iter_data in enumerate(iterations_data, 1):
            if "error" in iter_data:
                continue

            engineer_info = iter_data.get("engineer", {})
            st.markdown(f"### Iteration {iter_num}")

            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Score", f"{engineer_info.get('best_score', 0):.4f}")
            with col2:
                fedot_score = engineer_info.get("fedot_result", {}).get("score", 0)
                st.metric("Fedot Score", f"{fedot_score:.4f}")
            with col3:
                st.metric("Best Model", engineer_info.get("best_model", "N/A"))

            # Baseline results table
            baseline_results = engineer_info.get("baseline_results", [])
            if baseline_results:
                st.markdown("**Baseline Pipeline Results:**")
                rows = []
                for br in baseline_results:
                    row = {"Pipeline": br.get("name", "N/A"), "Score": br.get("score", 0)}
                    metrics = br.get("metrics", {})
                    row.update(metrics)
                    if br.get("error"):
                        row["Error"] = br["error"]
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Fedot config used
            fedot_config_used = engineer_info.get("fedot_config_used", {})
            if fedot_config_used:
                render_fedot_config(fedot_config_used, "Fedot Config Used")

            # Comparison
            comparison = engineer_info.get("comparison", {})
            if comparison:
                st.markdown("**Comparison:**")
                st.write(comparison.get("comparison_text", ""))
                if comparison.get("architect_wins"):
                    st.success("Architect baselines won!")
                else:
                    st.info("Fedot.Industrial won!")

                # Cross-iteration comparison
                if comparison.get("cross_iteration_text"):
                    st.markdown("**Cross-iteration progress:**")
                    improved = comparison.get("improved_over_previous", False)
                    if improved:
                        st.success(comparison["cross_iteration_text"])
                    else:
                        st.warning(comparison["cross_iteration_text"])

            # Tool calls
            render_tool_calls(engineer_info.get("tool_calls", []), "Engineer Tool Calls")

            st.divider()
    else:
        render_agent_card("engineer")


# ============== CRITIC TAB ==============

with tab_critic:
    st.subheader("Critic: Evaluation & Feedback")

    if "result" in st.session_state:
        result = st.session_state.result
        iterations_data = result.get("iterations", [])

        for iter_num, iter_data in enumerate(iterations_data, 1):
            if "error" in iter_data:
                continue

            critic_info = iter_data.get("critic", {})
            st.markdown(f"### Iteration {iter_num}")

            # Winner and stop
            col1, col2, col3 = st.columns(3)
            with col1:
                winner = critic_info.get("winner", "N/A").upper()
                if winner == "ARCHITECT":
                    st.success(f"Winner: {winner}")
                elif winner == "FEDOT":
                    st.info(f"Winner: {winner}")
                else:
                    st.write(f"Winner: {winner}")
            with col2:
                if critic_info.get("should_stop"):
                    st.warning("Recommends STOP")
                else:
                    st.write("Continue iterating")
            with col3:
                st.write(f"Assessment: {critic_info.get('score_assessment', 'N/A')[:100]}")

            # Strengths/Weaknesses
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Strengths:**")
                for s in critic_info.get("strengths", []):
                    st.write(f"+ {s}")
            with col2:
                st.markdown("**Weaknesses:**")
                for w in critic_info.get("weaknesses", []):
                    st.write(f"- {w}")

            # Suggested Fedot changes
            fedot_changes = critic_info.get("suggested_fedot_changes", {})
            if fedot_changes:
                st.markdown("**Suggested Fedot Config Changes for Next Iteration:**")
                change_cols = st.columns(min(len(fedot_changes), 4))
                for i, (k, v) in enumerate(fedot_changes.items()):
                    with change_cols[i % len(change_cols)]:
                        st.metric(k, str(v))

            # Pipeline changes
            pipeline_changes = critic_info.get("suggested_pipeline_changes", [])
            if pipeline_changes:
                st.markdown("**Suggested Pipeline Changes:**")
                for pc in pipeline_changes:
                    st.write(f"- {pc}")

            # Error analysis
            error_analysis = critic_info.get("error_analysis", {})
            if error_analysis:
                with st.expander("Error Analysis Details"):
                    st.json(error_analysis)

            # Feature importance
            feature_imp = critic_info.get("feature_importance", {})
            if feature_imp:
                with st.expander("Feature Importance"):
                    st.json(feature_imp)

            # Explanation (from Fedot explain)
            explanation = critic_info.get("explanation", {})
            if explanation:
                with st.expander("Model Explanation (Fedot.Industrial)"):
                    st.json(explanation)

            # Tool calls
            render_tool_calls(critic_info.get("tool_calls", []), "Critic Tool Calls")

            with st.expander("Full LLM Response"):
                st.code(critic_info.get("full_response", "N/A"), language="text")

            st.divider()
    else:
        render_agent_card("critic")


# ============== REPORT TAB (Scribe) ==============

with tab_report:
    st.subheader("Final Report (Scribe)")

    if "result" in st.session_state:
        result = st.session_state.result
        report = result.get("report", {})

        if report and not report.get("error"):
            st.markdown(f"## {report.get('title', 'ML Experiment Report')}")

            if report.get("summary"):
                st.markdown("### Summary")
                st.write(report["summary"])

            if report.get("methodology"):
                st.markdown("### Methodology")
                st.write(report["methodology"])

            if report.get("results"):
                st.markdown("### Results")
                st.write(report["results"])

            if report.get("recommendations"):
                st.markdown("### Recommendations")
                for rec in report["recommendations"]:
                    st.write(f"- {rec}")

            # Tool calls
            render_tool_calls(report.get("tool_calls", []), "Scribe Tool Calls")

            with st.expander("Full LLM Response"):
                st.code(report.get("full_response", "N/A"), language="text")
        elif report.get("error"):
            st.error(f"Report generation failed: {report['error']}")
        else:
            st.info("No report generated")
    else:
        render_agent_card("scribe")


# ============== SUMMARY TAB ==============

with tab_summary:
    st.subheader("Summary & Iteration Comparison")

    if "result" in st.session_state:
        result = st.session_state.result
        iterations_data = result.get("iterations", [])
        summary = result.get("summary", {})

        # Task type info
        if result.get("is_time_series"):
            st.info(
                f"Time Series Task: **{result.get('task_type')}**"
                + (f" | Forecast Length: **{result.get('forecast_length')}**" if result.get("forecast_length") else "")
            )

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Iterations", summary.get("total_iterations", len(iterations_data)))
        with col2:
            st.metric("Successful", summary.get("successful_iterations", 0))
        with col3:
            st.metric("Max Architect Score", f"{summary.get('max_architect_score', 0):.4f}")
        with col4:
            st.metric("Max Fedot Score", f"{summary.get('max_fedot_score', 0):.4f}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Architect Score", f"{summary.get('avg_architect_score', 0):.4f}")
        with col2:
            st.metric("Avg Fedot Score", f"{summary.get('avg_fedot_score', 0):.4f}")
        with col3:
            if summary.get("early_stopped"):
                st.warning("Early stopped by Critic")
            else:
                st.write(f"Architect wins: {summary.get('architect_wins_count', 0)}")

        # Iteration comparison table
        if iterations_data:
            st.subheader("Iteration Comparison")
            rows = []
            for iter_data in iterations_data:
                if "error" in iter_data:
                    rows.append({
                        "Iteration": iter_data.get("iteration", "?"),
                        "Status": "FAILED",
                        "Error": iter_data.get("error", ""),
                    })
                else:
                    eng = iter_data.get("engineer", {})
                    crit = iter_data.get("critic", {})
                    arch = iter_data.get("architect", {})
                    rows.append({
                        "Iteration": iter_data.get("iteration", "?"),
                        "Best Score": eng.get("best_score", 0),
                        "Best Model": eng.get("best_model", "N/A"),
                        "Fedot Score": eng.get("fedot_result", {}).get("score", 0),
                        "Fedot Preset": arch.get("fedot_config", {}).get("preset", "N/A"),
                        "Winner": crit.get("winner", "N/A"),
                        "Stop": crit.get("should_stop", False),
                    })

            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Errors
        error_logs = []
        for i, iter_data in enumerate(iterations_data, 1):
            if "error" in iter_data:
                error_logs.append({"Iteration": i, "Error": iter_data["error"]})

        if error_logs:
            st.subheader("Errors")
            st.dataframe(pd.DataFrame(error_logs), use_container_width=True)
        else:
            st.success("No errors!")

        # Best iteration details
        best = result.get("best_iteration", {})
        if best and "error" not in best:
            with st.expander("Best Iteration Details"):
                st.json(best)
    else:
        st.info("Run orchestration to see summary")
