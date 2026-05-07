"""Architect: asks the LLM to synthesize validated pipeline graphs."""

import json
import logging
from typing import Any, Dict, Optional

from .base_agent import BaseAgent, extract_json_block
from .schemas import ArchitectResult, CriticFeedback, DataContext
from .structured import parse_graph_proposal_object

logger = logging.getLogger("Architect")


class Architect(BaseAgent):
    ALLOWED_TOOLS = [
        "get_data_profile",
        "get_available_operations",
        "propose_graph",
        "mutate_graph",
        "visualize_graph",
    ]

    SYSTEM_PROMPT = """You are an ML Architect for MultiAgentFedot.IndustrialSystem (MAFIS).
You design Fedot.Industrial pipeline graphs from the provided data profile,
operation catalog, previous graph, and Critic feedback.

Return only one strict JSON object:
{
  "graph": {
    "task_type": "classification",
    "nodes": [
      {"id": "model", "operation": "operation_from_catalog", "inputs": []}
    ]
  },
  "analysis": "why this graph fits the evidence",
  "reasoning": "why these operations and connections were selected from the catalog"
}

Use only operations listed in available_operations. Every input must be an
existing node id, or [] for raw data. The graph must be a DAG with exactly one
root model node. The graph passed to Fedot.Industrial as `initial_assumption` is
finetuned by AutoML. Do not propose node parameters, model hyperparameters, or
operation search-space values; node params must be omitted or empty and are
ignored by the backend. Industrial strategies
(default/federated_automl/sampling_strategy) are execution modes selected
outside the graph; do not include them in the JSON. The default mode means
Fedot.Industrial's default strategy with task data_type supplied by config.
For tabular datasets, the backend prepares numeric feature values before handing
the native (X, y) tuple to Fedot.Industrial; use preprocessing nodes only when
they are valid operations from the catalog.
Return JSON only."""

    STRUCTURED_PROMPT = SYSTEM_PROMPT

    def __init__(self, name: str = "Architect", mcp_client=None):
        super().__init__(name=name, mcp_client=mcp_client)

    async def execute(
        self,
        data_context: DataContext,
        iteration: int,
        prev_feedback: Optional[CriticFeedback] = None,
        prev_graph: Optional[dict] = None,
    ) -> ArchitectResult:
        del iteration
        self._tool_call_log = []
        result = ArchitectResult()

        try:
            graph, mermaid, analysis, reasoning, diagnostics = await self._structured_graph_proposal(
                data_context,
                prev_feedback,
                prev_graph,
            )
            result.diagnostics.extend(diagnostics)
            if graph:
                result.graph = graph
                result.mermaid = mermaid
                result.analysis = analysis
                result.reasoning = reasoning
            else:
                logger.info("[Architect] LLM did not produce a valid graph for %s", data_context.task_type)
            result.tool_calls = self.get_tool_calls()
            return result

        except Exception as exc:
            logger.exception("[Architect] error")
            result.diagnostics.append(
                {
                    "agent": "Architect",
                    "kind": "architect_error",
                    "summary": "Architect could not complete the graph proposal flow.",
                    "technical_message": str(exc),
                    "recommendations": ["Ask Architect to retry with a shorter instruction or edit an existing approved graph."],
                    "recoverable": True,
                }
            )
            result.tool_calls = self.get_tool_calls()
            return result

    async def _structured_graph_proposal(self, data_context: DataContext, prev_feedback, prev_graph):
        diagnostics = []
        profile = await self.call_mcp_tool(
            "get_data_profile",
            {
                "csv_path": data_context.csv_path,
                "target_column": data_context.target_column,
                "task_type": data_context.task_type,
            },
        )
        operations = await self.call_mcp_tool("get_available_operations", {"task_type": data_context.task_type})
        operations = self._structure_only_operations(operations)

        repair_notes = ""
        last_analysis = ""
        last_reasoning = ""
        for attempt in range(2):
            prompt = self._build_structured_prompt(data_context, profile, operations, prev_feedback, prev_graph)
            if repair_notes:
                prompt += (
                    "\nThe previous response was rejected by validation. "
                    "Return a corrected GraphProposal JSON object only.\n"
                    f"Validation feedback: {repair_notes}"
                )
            response = await self.call_llm(
                prompt,
                system_prompt=self.STRUCTURED_PROMPT,
                max_rounds=1,
                use_tools=False,
            )
            raw_text = response.get("full_response", "") or response.get("error", "")
            parsed = extract_json_block(raw_text)
            proposal = parse_graph_proposal_object(parsed) if parsed else None
            if not proposal:
                repair_notes = raw_text[:1200] or "No parseable JSON object."
                if attempt == 1:
                    diagnostics.append(
                        {
                            "agent": "Architect",
                            "kind": "invalid_structured_llm_output",
                            "summary": "Architect LLM response was not a valid GraphProposal JSON object.",
                            "technical_message": repair_notes,
                            "recommendations": [
                                "Return strict JSON: use null instead of Python None, true/false instead of True/False."
                            ],
                            "recoverable": True,
                        }
                    )
                continue

            last_analysis = proposal.analysis
            last_reasoning = proposal.reasoning

            proposed = await self.call_mcp_tool("propose_graph", {"graph_json": proposal.graph.as_graph_json()})
            if isinstance(proposed, dict) and proposed.get("valid") and proposed.get("graph"):
                return (
                    proposed["graph"],
                    proposed.get("mermaid", ""),
                    proposal.analysis,
                    proposal.reasoning,
                    diagnostics,
                )

            repair_notes = json.dumps(proposed, ensure_ascii=False)[:1200]
            if attempt == 1:
                diagnostics.append(
                    {
                        "agent": "Architect",
                        "kind": "invalid_structured_graph",
                        "summary": "Architect proposed a structured graph, but graph validation rejected it.",
                        "technical_message": repair_notes,
                        "recommendations": [
                            "Use operations exactly as returned by the operation catalog.",
                            "Every node input must be an existing graph.nodes id; use inputs=[] for raw data.",
                            "Connect nodes by their actual id values rather than generated input/output handle names.",
                        ],
                        "recoverable": True,
                    }
                )
        return None, "", last_analysis, last_reasoning, diagnostics

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
            "industrial_strategy_in_use": dc.industrial_strategy or "default",
            "industrial_strategy_params": dict(dc.industrial_strategy_params or {}),
        }
        if prev_fb:
            payload["feedback"] = {
                "decision": prev_fb.winner,
                "assessment": prev_fb.assessment,
                "strengths": prev_fb.strengths,
                "weaknesses": prev_fb.weaknesses,
                "suggested_mutations": prev_fb.suggested_mutations,
                "improvement_plan": prev_fb.improvement_plan,
                "questions_for_user": prev_fb.questions_for_user,
                "diagnostics": prev_fb.diagnostics,
            }
        return (
            "Create one valid graph proposal for this payload.\n"
            "The graph you produce becomes Fedot.Industrial `initial_assumption` and is polished by AutoML finetune.\n"
            "Choose only operations and graph topology. Do not choose hyperparameters or node params; "
            "Fedot.Industrial owns parameter tuning during finetune.\n"
            "For tabular data, the backend passes a native Fedot.Industrial (X, y) tuple "
            "with numeric feature values; do not rely on manual InputData metadata.\n"
            "available_operations.industrial_templates contains Fedot.Industrial-native graph patterns.\n"
            "available_operations.industrial_strategies_catalog describes selectable execution strategies "
            "(default/federated_automl/sampling_strategy). Strategies are picked by the user outside the graph; "
            "do not include them in your JSON.\n"
            "For graph node inputs, use [] for raw data and otherwise use only existing node ids.\n"
            "Return strict JSON literals: null/true/false, not Python None/True/False.\n"
            "Return only the GraphProposal JSON object.\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    @staticmethod
    def _structure_only_operations(operations: Dict[str, Any]) -> Dict[str, Any]:
        """Remove parameter/search-space hints before sending the catalog to Architect."""
        if not isinstance(operations, dict):
            return {}
        cleaned = dict(operations)
        catalog = cleaned.get("catalog")
        if isinstance(catalog, dict):
            cleaned_catalog: Dict[str, Any] = {}
            for group, items in catalog.items():
                if not isinstance(items, list):
                    cleaned_catalog[group] = items
                    continue
                cleaned_catalog[group] = [
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"param_hints", "industrial_search_space"}
                    }
                    for item in items
                    if isinstance(item, dict)
                ]
            cleaned["catalog"] = cleaned_catalog
        if isinstance(cleaned.get("default_graph"), list):
            cleaned["default_graph"] = [
                {key: value for key, value in node.items() if key != "params"}
                for node in cleaned["default_graph"]
                if isinstance(node, dict)
            ]
        if isinstance(cleaned.get("industrial_templates"), list):
            templates = []
            for template in cleaned["industrial_templates"]:
                if not isinstance(template, dict):
                    continue
                item = dict(template)
                graph = item.get("graph")
                if isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
                    item["graph"] = {
                        **graph,
                        "nodes": [
                            {key: value for key, value in node.items() if key != "params"}
                            for node in graph["nodes"]
                            if isinstance(node, dict)
                        ],
                    }
                templates.append(item)
            cleaned["industrial_templates"] = templates
        return cleaned
