import streamlit as st
import requests
import pandas as pd
import json
import os
from io import StringIO

st.set_page_config(page_title="Multi-agent system", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8001")

st.title("Multi-agent system with Fedot.Industrial")

tab_data, tab_architect, tab_engineer, tab_critic, tab_describe, tab_summary = st.tabs([
    "Data", "Architect", "Engineer", "Critic", "Describe", "Summary"
])

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
            key="target_select"
        )
        
        task_type = st.radio(
            "Task Type",
            options=["regression", "classification"],
            key="task_select"
        )
        
        iterations = st.slider(
            "Iterations",
            min_value=1,
            max_value=5,
            value=2,
            key="iterations_slider"
        )
        
        run_button = st.button("Run Orchestration", key="run_orchestration")
        
        if run_button:
            with st.spinner("Running orchestration..."):
                try:
                    unique_name = f"uploaded_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    shared_path = os.path.join("/app/data", unique_name)
                    
                    df.to_csv(shared_path, index=False)
                    
                    payload = {
                        "csv_path": shared_path,
                        "target_column": target_column,
                        "task_type": task_type,
                        "iterations": iterations,
                        "fedot_url": "http://fedot-server:8000",
                    }
                    
                    url = f"{BACKEND_URL}/orchestrate"
                    response = requests.post(url, json=payload, timeout=600)
                    response.raise_for_status()
                    
                    result = response.json()
                    st.session_state.result = result
                    
                    with st.container():
                        st.success("Orchestration completed successfully!")
                        st.info(f"Total iterations completed: {len(result.get('iterations', []))}")
                
                except requests.Timeout:
                    st.error("⏱Request timeout - orchestration may still be running on backend")
                except Exception as e:
                    st.error(f"Error: {e}")

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
        
        st.write("Data Types:")
        st.dataframe(df.dtypes, use_container_width=True)
        
        st.write("Statistics:")
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.info("Upload a CSV file to see data preview")

with tab_architect:
    st.subheader("Architect Analysis and Pipeline Selection")
    
    if "result" in st.session_state:
        result = st.session_state.result
        iterations_data = result.get("iterations", [])
        
        if iterations_data:
            for iter_num, iter_data in enumerate(iterations_data, 1):
                if "error" not in iter_data:
                    architect_info = iter_data.get("architect", {})
                    
                    st.markdown(f"### Iteration {iter_num}")
                    
                    # Show analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Task Identified:**")
                        st.info(architect_info.get("task", "Unknown"))
                        
                        st.markdown("**Selected Pipelines:**")
                        selected = architect_info.get("selected_pipelines", [])
                        for pipeline in selected:
                            st.write(f"{pipeline}")
                    
                    with col2:
                        st.markdown("**Analysis Summary:**")
                        st.info(architect_info.get("analysis", "No analysis"))
                        
                        st.markdown("**Expected Improvement:**")
                        st.success(architect_info.get("expected", "Unknown"))
                    
                    st.markdown("**Reasoning:**")
                    st.write(architect_info.get("reasoning", "No reasoning provided"))
                    
                    with st.expander("Full LLM Response"):
                        full_response = architect_info.get("full_response", "No response available")
                        st.code(full_response, language="text")
                    
                    if "error" in architect_info:
                        st.error(f"Error: {architect_info['error']}")
                    
                    break
            else:
                st.error("No successful architect analyses found")
                for i, iter_data in enumerate(iterations_data, 1):
                    if "error" in iter_data:
                        st.warning(f"Iteration {i}: {iter_data['error']}")
        else:
            st.info("No iterations completed yet")
    else:
        st.info("Run orchestration to see architect analysis")

with tab_engineer:
    st.subheader("Engineer Training Results")
    
    if "result" in st.session_state:
        result = st.session_state.result
        iterations_data = result.get("iterations", [])
        
        if iterations_data:
            for iter_data in iterations_data:
                if "error" not in iter_data:
                    engineer_info = iter_data.get("engineer", {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Architect Score", f"{engineer_info.get('best_score', 0):.4f}")
                    with col2:
                        st.metric("Fedot Score", f"{engineer_info.get('fedot_score', 0):.4f}")
                    with col3:
                        st.metric("Best Model", engineer_info.get('best_model', 'N/A'))
                    
                    if engineer_info.get("failed_pipelines"):
                        st.warning("Failed Pipelines:")
                        for pipeline_name, error in engineer_info.get("failed_pipelines", {}).items():
                            st.error(f"- {pipeline_name}: {error}")
                    
                    st.subheader("Architect vs Fedot Comparison")
                    comparison = engineer_info.get("comparison", {})
                    if comparison:
                        st.json(comparison)
                    else:
                        st.write("No comparison data")
                    
                    break
        else:
            st.warning("No successful engineer results")
    else:
        st.info("Run orchestration to see engineer results")

with tab_critic:
    st.subheader("Critic Analysis")
    
    if "result" in st.session_state:
        result = st.session_state.result
        iterations_data = result.get("iterations", [])
        
        if iterations_data:
            for iter_data in iterations_data:
                if "error" not in iter_data:
                    critic_info = iter_data.get("critic", {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Winner:**")
                        winner = critic_info.get('winner', 'N/A').upper()
                        if winner == "ARCHITECT":
                            st.success(winner)
                        elif winner == "FEDOT":
                            st.info(winner)
                        else:
                            st.write(winner)
                    
                    with col2:
                        st.markdown("**Feedback:**")
                        st.info(critic_info.get("feedback", "No feedback"))
                    
                    st.markdown("**Analysis:**")
                    st.write(critic_info.get("analysis", "No analysis"))
                    
                    st.markdown("**Strengths:**")
                    strengths = critic_info.get("strengths", [])
                    for strength in strengths:
                        st.write(f"✓ {strength}")
                    
                    st.markdown("**Weaknesses:**")
                    weaknesses = critic_info.get("weaknesses", [])
                    for weakness in weaknesses:
                        st.write(f"✗ {weakness}")
                    
                    with st.expander("Full LLM Response"):
                        full_response = critic_info.get("full_response", "No response available")
                        st.code(full_response, language="text")
                    
                    if "error" in critic_info:
                        st.error(f"Error: {critic_info['error']}")
                    
                    break
        else:
            st.warning("No successful critic results")
    else:
        st.info("Run orchestration to see critic analysis")

with tab_describe:
    st.subheader("Solution Explanation")
    
    if "result" in st.session_state:
        result = st.session_state.result
        iterations_data = result.get("iterations", [])
        
        if iterations_data:
            for iter_data in iterations_data:
                if "error" not in iter_data:
                    describe_info = iter_data.get("describe", {})
                    
                    st.markdown(f"### {describe_info.get('title', 'ML Solution')}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Architecture:**")
                        st.write(describe_info.get("architecture", "N/A"))
                        
                        st.markdown("**Components:**")
                        components = describe_info.get("components", [])
                        for comp in components:
                            st.write(f"- {comp}")
                    
                    with col2:
                        st.markdown("**Performance:**")
                        st.write(describe_info.get("performance", "N/A"))
                        
                        st.markdown("**vs Fedot:**")
                        st.write(describe_info.get("vs_fedot", "N/A"))
                    
                    st.markdown("**Recommendations:**")
                    recommendations = describe_info.get("recommendations", [])
                    for rec in recommendations:
                        st.write(f"• {rec}")
                    
                    with st.expander("Full LLM Response"):
                        full_response = describe_info.get("full_response", "No response available")
                        st.code(full_response, language="text")
                    
                    if "error" in describe_info:
                        st.error(f"Error: {describe_info['error']}")
                    
                    break
        else:
            st.warning("No successful describe results")
    else:
        st.info("Run orchestration to see explanation")

with tab_summary:
    st.subheader("Summary and Errors Log")
    
    if "result" in st.session_state:
        result = st.session_state.result
        iterations_data = result.get("iterations", [])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Iterations", len(iterations_data))
        with col2:
            successful = sum(1 for i in iterations_data if "error" not in i)
            st.metric("Successful", successful)
        with col3:
            failed = sum(1 for i in iterations_data if "error" in i)
            st.metric("Failed", failed)
        
        st.subheader("Errors and Issues")
        error_logs = []
        
        for i, iter_data in enumerate(iterations_data):
            iteration_num = i + 1
            
            if "error" in iter_data:
                error_logs.append({
                    "Iteration": iteration_num,
                    "Step": "General",
                    "Status": "FAILED",
                    "Error": iter_data["error"]
                })
            
            for step in iter_data.get("steps", []):
                if "error" in step:
                    error_logs.append({
                        "Iteration": iteration_num,
                        "Step": step.get("name", "Unknown"),
                        "Status": "FAILED",
                        "Error": step["error"]
                    })
                elif step.get("failed_pipelines"):
                    for pipeline, error in step["failed_pipelines"].items():
                        error_logs.append({
                            "Iteration": iteration_num,
                            "Step": step.get("name", "Unknown"),
                            "Status": "PARTIAL",
                            "Error": f"{pipeline}: {error}"
                        })
        
        if error_logs:
            error_df = pd.DataFrame(error_logs)
            st.dataframe(error_df, use_container_width=True)
            
            with st.expander("Full Error Details"):
                for idx, error_log in enumerate(error_logs):
                    st.write(f"Iteration {error_log['Iteration']} - {error_log['Step']}")
                    st.code(error_log["Error"], language="text")
        else:
            st.success("No errors found!")
        
        st.subheader("Iteration Details")
        for i, iter_data in enumerate(iterations_data):
            with st.expander(f"Iteration {i+1}", expanded=False):
                if "error" in iter_data:
                    st.error(f"FAILED: {iter_data['error']}")
                else:
                    steps = iter_data.get("steps", [])
                    for step in steps:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            status = step.get("status", "UNKNOWN")
                            st.write(f"**{step.get('name', 'Unknown')}** {status}")
                        with col2:
                            if "error" in step:
                                st.error(step["error"])
                            elif step.get("failed_pipelines"):
                                for p, e in step["failed_pipelines"].items():
                                    st.warning(f"{p}: {e}")
                    
                    if not steps:
                        st.write("No steps data")
    else:
        st.info("Run orchestration to see summary and errors")