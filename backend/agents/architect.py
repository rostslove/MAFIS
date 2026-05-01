"""Architect: synthesizes a pipeline graph for the task using MCP tools."""

import json
import logging
from typing import Optional

from .base_agent import BaseAgent, extract_json_block
from .schemas import ArchitectResult, CriticFeedback, DataContext
from .structured import parse_graph_proposal_object

logger = logging.getLogger("Architect")


class Architect(BaseAgent):
    ALLOWED_TOOLS = [
        "get_data_profile", "get_available_operations",
        "propose_graph", "mutate_graph", "visualize_graph",
    ]

    SYSTEM_PROMPT = """You are an ML Architect. You design pipeline GRAPHS by composing atomic operations.

If native tool calling is unavailable, request a tool by returning exactly one JSON object and no prose:
{"name": "tool_name", "arguments": {"arg": "value"}}

A graph is JSON: {"task_type": "...", "nodes": [{"id": "model", "operation": "operation_from_catalog", "params": {}, "inputs": []}]}.

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

Use only operations returned by get_available_operations. For tabular classification, operations like
fourier_basis, quantile_extractor, industrial_freq_clf, fft_features, and statistical_extraction are invalid
unless the selected task is a time-series task and the operation is in the returned catalog.
"""

    STRUCTURED_PROMPT = """You are an ML Architect. Return JSON only, no markdown and no prose.
The response must match this schema:
{
  "graph": {
    "task_type": "classification",
    "nodes": [
      {"id": "model", "operation": "rf", "params": {}, "inputs": []}
    ]
  },
  "analysis": "Short explanation of why this graph fits the data and task.",
  "reasoning": "Short explanation of operation choices using only the available operation catalog."
}
Rules:
- Use only operations listed in AVAILABLE_OPERATIONS.
- If AVAILABLE_OPERATIONS.preprocessing is empty, do not add preprocessing nodes.
- The graph must have exactly one root model node.
- For ordinary tabular classification/regression, prefer one direct model node unless feedback explicitly asks for another valid model or explicit model parameters.
- If feedback contains suggested_mutations, reflect them in the returned graph. Do not return the previous graph unchanged.
- If feedback mentions class imbalance and param hints allow it, use explicit balancing params such as xgboost.scale_pos_weight or class_weight="balanced".
- If feedback mentions unstable cross-validation, use a simpler model or explicit regularization params from param_hints.
- If data_profile.benchmark is "M4", create a ts_classification graph for classifying frequency groups from fixed-length time-series windows.
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
            graph, mermaid, analysis, reasoning, diagnostics = await self._structured_graph_proposal(
                data_context,
                prev_feedback,
                prev_graph,
            )

            if not graph:
                result.diagnostics.extend(diagnostics)
                if self._requires_explicit_architect_graph(data_context):
                    result.diagnostics.append(
                        {
                            "agent": "Architect",
                            "kind": "architect_required_for_benchmark",
                            "summary": "Architect did not produce a valid benchmark graph.",
                            "technical_message": "M4 benchmark mode does not use a deterministic default graph.",
                            "recommendations": [
                                "Retry with a shorter request.",
                                "Mention exact ts_classification operations from the catalog.",
                                "Use the graph editor to create and approve a benchmark graph manually.",
                            ],
                            "recoverable": True,
                        }
                    )
                    result.tool_calls = self.get_tool_calls()
                    return result
                graph, mermaid = await self._fallback_graph_with_tools(data_context, prev_graph)
                analysis = self._fallback_analysis(data_context.task_type, bool(prev_graph))
                reasoning = self._fallback_reasoning(graph, data_context.task_type)
                result.diagnostics.append(
                    {
                        "agent": "Architect",
                        "kind": "structured_proposal_invalid",
                        "summary": "Architect could not produce a valid structured graph proposal, so a deterministic MCP-backed proposal was used.",
                        "technical_message": json.dumps(diagnostics, ensure_ascii=False)[:1200],
                        "recommendations": [
                            "Approve the deterministic graph if it is suitable, or edit it manually before evaluation.",
                            "For local 7B models, keep graph requests short and operation names exact.",
                        ],
                        "recoverable": True,
                    }
                )
                logger.info("[Architect] Using fallback graph for %s", data_context.task_type)

            result.graph = graph
            result.mermaid = mermaid
            result.analysis = analysis or self._fallback_analysis(data_context.task_type, bool(prev_graph))
            result.reasoning = reasoning or self._fallback_reasoning(graph, data_context.task_type)
            result.tool_calls = self.get_tool_calls()
            return result

        except Exception as e:
            logger.exception(f"[Architect] error")
            if self._requires_explicit_architect_graph(data_context):
                result.analysis = f"Architect failed: {e}"
                result.diagnostics.append(
                    {
                        "agent": "Architect",
                        "kind": "architect_error",
                        "summary": "Architect could not complete the benchmark graph proposal flow.",
                        "technical_message": str(e),
                        "recommendations": ["Retry or create a benchmark graph manually in the editor."],
                        "recoverable": True,
                    }
                )
                result.tool_calls = self.get_tool_calls()
                return result
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

    async def _structured_graph_proposal(self, data_context: DataContext, prev_feedback, prev_graph):
        diagnostics = []
        if data_context.profile.get("benchmark"):
            profile = data_context.profile
        else:
            profile = await self.call_mcp_tool(
                "get_data_profile",
                {
                    "csv_path": data_context.csv_path,
                    "target_column": data_context.target_column,
                    "task_type": data_context.task_type,
                },
            )
        operations = await self.call_mcp_tool("get_available_operations", {"task_type": data_context.task_type})

        prompt = self._build_structured_prompt(data_context, profile, operations, prev_feedback, prev_graph)
        response = await self.call_llm(
            prompt,
            system_prompt=self.STRUCTURED_PROMPT,
            max_rounds=1,
            use_tools=False,
        )
        raw_text = response.get("full_response", "")
        parsed = extract_json_block(raw_text)
        proposal = parse_graph_proposal_object(parsed) if parsed else None
        if not proposal:
            diagnostics.append(
                {
                    "agent": "Architect",
                    "kind": "invalid_structured_llm_output",
                    "summary": "Architect LLM response was not a valid GraphProposal JSON object.",
                    "technical_message": raw_text[:1200] or response.get("error", ""),
                    "recommendations": ["The backend will use a deterministic graph proposal instead."],
                    "recoverable": True,
                }
            )
            return None, "", "", "", diagnostics

        proposed = await self.call_mcp_tool("propose_graph", {"graph_json": proposal.graph.as_graph_json()})
        if isinstance(proposed, dict) and proposed.get("valid") and proposed.get("graph"):
            return (
                proposed["graph"],
                proposed.get("mermaid", ""),
                proposal.analysis,
                proposal.reasoning,
                diagnostics,
            )

        diagnostics.append(
            {
                "agent": "Architect",
                "kind": "invalid_structured_graph",
                "summary": "Architect proposed a structured graph, but graph validation rejected it.",
                "technical_message": json.dumps(proposed, ensure_ascii=False)[:1200],
                "recommendations": ["Use operations exactly as returned by the operation catalog."],
                "recoverable": True,
            }
        )
        return None, "", proposal.analysis, proposal.reasoning, diagnostics

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

    async def _fallback_graph_with_tools(self, data_context: DataContext, prev_graph: Optional[dict] = None):
        if not data_context.profile.get("benchmark"):
            await self.call_mcp_tool(
                "get_data_profile",
                {
                    "csv_path": data_context.csv_path,
                    "target_column": data_context.target_column,
                    "task_type": data_context.task_type,
                },
            )
        await self.call_mcp_tool("get_available_operations", {"task_type": data_context.task_type})

        graph, mermaid = self._fallback_graph(data_context.task_type, prev_graph)
        proposed = await self.call_mcp_tool(
            "propose_graph",
            {"graph_json": json.dumps(graph, ensure_ascii=False)},
        )
        if isinstance(proposed, dict) and proposed.get("valid") and proposed.get("graph"):
            return proposed["graph"], proposed.get("mermaid", mermaid)
        return graph, mermaid

    def _build_user_message(self, dc: DataContext, iteration: int, prev_fb, prev_graph) -> str:
        profile = dc.profile
        msg = (
            f"Evaluation request: {iteration}\n"
            f"Task: {dc.task_type} (TS: {dc.is_time_series})\n"
            f"CSV: {dc.csv_path}\n"
            f"Target: {dc.target_column}\n"
            f"Primary metric: {dc.primary_metric or 'default'}\n"
            f"Profile: {profile.get('n_samples')} samples x {profile.get('n_features')} features\n"
            f"Issues: {profile.get('issues', [])}\n"
        )
        if dc.forecast_length:
            msg += f"Forecast length: {dc.forecast_length}\n"

        if dc.iteration_history:
            msg += "\nPREVIOUS APPROVED EVALUATIONS:\n"
            for rec in dc.iteration_history:
                msg += f"  Eval {rec.iteration}: graph_score={rec.graph_score:.4f}, decision={rec.winner}\n"
            msg += "Avoid repeating approaches that did not address Critic feedback. Build on what worked.\n"

        if prev_graph:
            msg += f"\nPREVIOUS GRAPH:\n{json.dumps(prev_graph)}\n"

        if prev_fb:
            msg += (
                f"\nFEEDBACK:\n"
                f"  decision: {prev_fb.winner}\n"
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

    def _build_structured_prompt(self, dc: DataContext, profile, operations, prev_fb, prev_graph) -> str:
        payload = {
            "task": dc.task_type,
            "is_time_series": dc.is_time_series,
            "csv_path": dc.csv_path,
            "target_column": dc.target_column,
            "primary_metric": dc.primary_metric,
            "data_profile": profile,
            "available_operations": operations,
            "previous_graph": prev_graph,
            "feedback": None,
        }
        if prev_fb:
            payload["feedback"] = {
                "decision": prev_fb.winner,
                "assessment": prev_fb.assessment,
                "weaknesses": prev_fb.weaknesses,
                "suggested_mutations": prev_fb.suggested_mutations,
                "improvement_plan": prev_fb.improvement_plan,
            }
        return (
            "Create one valid graph proposal for this payload.\n"
            "Use available_operations.catalog[].param_hints when setting params.\n"
            "Use available_operations.training_strategies as training guidance, but encode the actionable change in graph nodes/params.\n"
            "Return only the GraphProposal JSON object.\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    @staticmethod
    def _requires_explicit_architect_graph(dc: DataContext) -> bool:
        return bool(dc.profile.get("requires_architect_graph"))

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
                "Critic will judge whether a more complex graph or explicit parameters are justified."
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
