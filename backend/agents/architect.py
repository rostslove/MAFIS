"""Architect: synthesizes a pipeline graph for the task using MCP tools."""

import json
import logging
from typing import Optional

from .base_agent import BaseAgent, extract_json_block
from .schemas import ArchitectResult, CriticFeedback, DataContext

logger = logging.getLogger("Architect")


class Architect(BaseAgent):
    ALLOWED_TOOLS = [
        "get_data_profile", "get_available_operations",
        "propose_graph", "mutate_graph", "visualize_graph",
    ]

    SYSTEM_PROMPT = """You are an ML Architect. You design pipeline GRAPHS by composing atomic operations.

A graph is JSON: {"task_type": "...", "nodes": [{"id": "n1", "operation": "fourier_basis", "params": {}, "inputs": []}, {"id": "n2", "operation": "industrial_freq_clf", "params": {}, "inputs": ["n1"]}]}.

Each node has:
  - id: unique string
  - operation: an atomic op (call get_available_operations to see valid ops for the task)
  - params: dict of hyperparameters (use {} to let Engineer tune)
  - inputs: list of upstream node IDs (empty = consumes raw data)

The graph must be a DAG with exactly ONE root (node nobody else inputs from). The root is the model.
For ordinary tabular classification/regression in this project, prefer a direct model-only graph unless the available operations explicitly list preprocessing for that task. Do not invent preprocessing operations.

WORKFLOW:
1. Call get_data_profile to understand the data.
2. Call get_available_operations(task_type) to see valid operations.
3. Build a graph as JSON, then call propose_graph(graph_json) to validate it.
4. If you have prior feedback with suggested_mutations, call mutate_graph for each one and finally re-propose.
5. After tools, output ANALYSIS and REASONING in plain text.

Think step-by-step (Chain-of-Thought):
"For ECG classification, FFT features matter, then statistical extraction, then a freq classifier:
 - n1: fourier_basis (frequency features) <- raw
 - n2: quantile_extractor (statistics from spectrum) <- n1
 - n3: industrial_freq_clf (model) <- n2
The root is n3."
"""

    def __init__(self, name: str = "Architect", mcp_client=None):
        super().__init__(name=name, mcp_client=mcp_client)

    async def execute(
        self,
        data_context: DataContext,
        iteration: int,
        prev_feedback: Optional[CriticFeedback] = None,
        prev_graph: Optional[dict] = None,
    ) -> ArchitectResult:
        self._tool_call_log = []
        result = ArchitectResult()

        try:
            user_msg = self._build_user_message(data_context, iteration, prev_feedback, prev_graph)
            response = await self.call_llm(user_msg)
            text = response.get("full_response", "")
            if not response.get("success", True):
                result.diagnostics.append(
                    {
                        "agent": "Architect",
                        "kind": "llm_unavailable",
                        "summary": "Architect LLM call failed; a deterministic fallback graph was used.",
                        "technical_message": response.get("error", ""),
                        "recommendations": [
                            "Retry when the OpenRouter provider is available.",
                            "Set LLM_MODEL to a paid or less rate-limited model if you need full Architect reasoning.",
                            "You can still edit and approve the fallback graph manually.",
                        ],
                        "recoverable": True,
                    }
                )

            # Find the latest valid graph in tool call log
            graph, mermaid = self._extract_graph_from_log()

            # If LLM didn't successfully propose, fall back to default
            if not graph:
                result.diagnostics.extend(self._extract_diagnostics_from_log())
                graph, mermaid = self._fallback_graph(data_context.task_type, prev_graph)
                logger.info("[Architect] Using fallback graph for %s", data_context.task_type)

            result.graph = graph
            result.mermaid = mermaid
            result.analysis = self._extract_section(text, "ANALYSIS") or text[:500] or self._fallback_analysis(data_context.task_type, bool(prev_graph))
            result.reasoning = self._extract_section(text, "REASONING") or self._fallback_reasoning(graph, data_context.task_type)
            result.tool_calls = self.get_tool_calls()
            return result

        except Exception as e:
            logger.exception(f"[Architect] error")
            result.graph, result.mermaid = self._fallback_graph(data_context.task_type, prev_graph)
            result.analysis = f"Fallback: {e}"
            result.reasoning = self._fallback_reasoning(result.graph, data_context.task_type)
            result.diagnostics.append(
                {
                    "agent": "Architect",
                    "kind": "architect_error",
                    "summary": "Architect could not complete the graph proposal flow.",
                    "technical_message": str(e),
                    "recommendations": ["Use the fallback graph or adjust it manually in the editor."],
                    "recoverable": True,
                }
            )
            result.tool_calls = self.get_tool_calls()
            return result

    @staticmethod
    def _fallback_graph(task_type: str, prev_graph: Optional[dict] = None):
        from graph_engine import PipelineGraph

        if prev_graph:
            try:
                graph = PipelineGraph.from_dict(prev_graph)
                ok, _ = graph.validate()
                if ok:
                    return graph.to_dict(), graph.to_mermaid()
            except Exception:
                pass
        default = PipelineGraph.default(task_type)
        return default.to_dict(), default.to_mermaid()

    def _build_user_message(self, dc: DataContext, iteration: int, prev_fb, prev_graph) -> str:
        profile = dc.profile
        msg = (
            f"Iteration: {iteration}\n"
            f"Task: {dc.task_type} (TS: {dc.is_time_series})\n"
            f"CSV: {dc.csv_path}\n"
            f"Target: {dc.target_column}\n"
            f"Profile: {profile.get('n_samples')} samples x {profile.get('n_features')} features\n"
            f"Issues: {profile.get('issues', [])}\n"
        )
        if dc.forecast_length:
            msg += f"Forecast length: {dc.forecast_length}\n"

        if dc.iteration_history:
            msg += "\nPREVIOUS ITERATIONS:\n"
            for rec in dc.iteration_history:
                msg += f"  Iter {rec.iteration}: graph_score={rec.graph_score:.4f}, baseline={rec.best_baseline_score:.4f}, winner={rec.winner}\n"
            msg += "Avoid repeating approaches that didn't improve. Build on what worked.\n"

        if prev_graph:
            msg += f"\nPREVIOUS GRAPH:\n{json.dumps(prev_graph)}\n"

        if prev_fb:
            msg += (
                f"\nFEEDBACK:\n"
                f"  source/winner: {prev_fb.winner}\n"
                f"  assessment: {prev_fb.assessment}\n"
                f"  weaknesses/user notes: {prev_fb.weaknesses}\n"
                f"  suggested_mutations: {json.dumps(prev_fb.suggested_mutations)}\n"
            )
            if prev_fb.suggested_mutations:
                msg += "Apply these mutations using the mutate_graph tool.\n"
            else:
                msg += "Use this feedback to propose a revised validated graph.\n"

        msg += "\nUse the tools to build a validated graph, then output ANALYSIS and REASONING.\n"
        return msg

    @staticmethod
    def _fallback_analysis(task_type: str, reused_previous: bool) -> str:
        if reused_previous:
            return "Architect reused the last valid graph because the LLM provider did not return a new proposal."
        if task_type in ("classification", "regression"):
            return (
                "Architect selected a conservative model-only graph for ordinary tabular data. "
                "This avoids Fedot.Industrial preprocessing operations that can fail on single-column slices."
            )
        if task_type == "ts_forecasting":
            return "Architect selected a compact forecasting graph with a smoothing step and an autoregressive model."
        return "Architect selected a compact time-series graph with feature extraction followed by an industrial model."

    @staticmethod
    def _fallback_reasoning(graph: dict, task_type: str) -> str:
        nodes = graph.get("nodes", [])
        operations = " -> ".join(n.get("operation", "") for n in nodes)
        if task_type in ("classification", "regression"):
            return (
                f"Graph structure: {operations}. For tabular CSV data, the first reliable candidate is a direct model node; "
                "baselines are trained separately to judge whether a more complex graph is justified."
            )
        return (
            f"Graph structure: {operations}. The graph first extracts or transforms signal structure and then sends it to "
            "a task-specific model node."
        )

    def _extract_graph_from_log(self):
        """Find the most recent successful propose_graph or mutate_graph result."""
        graph, mermaid = None, ""
        for tc in self._tool_call_log:
            if tc.tool_name in ("propose_graph", "mutate_graph") and tc.success:
                r = tc.result
                if isinstance(r, dict) and r.get("valid") and r.get("graph"):
                    graph = r["graph"]
                    mermaid = r.get("mermaid", "")
        return graph, mermaid

    def _extract_diagnostics_from_log(self):
        diagnostics = []
        for tc in self._tool_call_log:
            payload = tc.result
            if isinstance(payload, dict):
                for item in payload.get("diagnostics", []) or []:
                    if isinstance(item, dict):
                        diagnostics.append({**item, "agent": item.get("agent") or "Architect"})
        return diagnostics

    @staticmethod
    def _extract_section(text: str, name: str) -> str:
        if not text:
            return ""
        idx = text.upper().find(name + ":")
        if idx == -1:
            return ""
        start = idx + len(name) + 1
        # Stop at next ALL CAPS heading or end of text
        end = len(text)
        for next_h in ("ANALYSIS:", "REASONING:", "WINNER:", "ASSESSMENT:", "STRENGTHS:", "WEAKNESSES:"):
            if next_h == name + ":":
                continue
            i = text.upper().find(next_h, start)
            if i != -1 and i < end:
                end = i
        return text[start:end].strip()
