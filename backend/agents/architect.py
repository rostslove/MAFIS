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
Fedot.Industrial's non-tabular-default Industrial repository unless diagnostics
explicitly say otherwise. For CSV feature matrices, the backend prepares numeric
feature values before handing the native (X, y) tuple to Fedot.Industrial.
Use runtime_contracts and fedot_industrial_meta_groups from available_operations
to keep preprocessing/model contracts compatible.
Apply senior ML judgement when choosing preprocessing. Consider the data
profile, model family assumptions, feature semantics, leakage risk,
dimensionality, sparsity, missingness, categorical structure, and whether a
transform changes the hypothesis space or merely adds complexity. Prefer a
compact graph whose preprocessing has a clear expected benefit for the chosen
downstream model. In the analysis and reasoning fields, justify each
preprocessing node by explaining the specific data/model interaction it
improves; omit preprocessing when that justification is weak.
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
            "data_profile": self._compact_profile(profile),
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
                "strengths": list(prev_fb.strengths or [])[:3],
                "weaknesses": list(prev_fb.weaknesses or [])[:3],
                "suggested_mutations": list(prev_fb.suggested_mutations or [])[:5],
                "improvement_plan": list(prev_fb.improvement_plan or [])[:5],
                "questions_for_user": list(prev_fb.questions_for_user or [])[:3],
                "diagnostics": list(prev_fb.diagnostics or [])[:5],
            }
        return (
            "Create one valid graph proposal for this payload.\n"
            "The graph you produce becomes Fedot.Industrial `initial_assumption` and is polished by AutoML finetune.\n"
            "Choose only operations and graph topology. Do not choose hyperparameters or node params; "
            "Fedot.Industrial owns parameter tuning during finetune.\n"
            "For plain CSV feature matrices, the backend normally passes a native Fedot.Industrial "
            "(X, y) tuple with numeric feature values; adapter-backed operations may instead be "
            "executed at the InputData/data-boundary layer before Fedot.Industrial finetune. "
            "Treat Fedot.Industrial repository operations as first-class candidates, and use "
            "runtime_contracts, runtime_adapters, and fedot_industrial_meta_groups from "
            "available_operations to avoid invalid contracts. Adapter-backed operations are allowed, "
            "but explain why their adapter semantics are useful for this graph.\n"
            "Act like an expert ML architect: choose preprocessing by reasoning about the data profile, "
            "model family assumptions, feature semantics, missingness, categorical structure, dimensionality, "
            "and whether the transform has a plausible benefit for the selected downstream model. "
            "Prefer compact graphs, and justify every preprocessing node in analysis/reasoning through its "
            "specific data/model interaction; omit transforms whose benefit is only generic or decorative.\n"
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
        """Keep the operation prompt compact enough for local Ollama contexts."""
        if not isinstance(operations, dict):
            return {}
        cleaned = {
            "task_type": operations.get("task_type"),
            "is_time_series": operations.get("is_time_series"),
            "preprocessing": list(operations.get("preprocessing", []) or []),
            "models": list(operations.get("models", []) or []),
            "metrics": list(operations.get("metrics", []) or []),
        }
        source_catalog = operations.get("catalog")
        if isinstance(source_catalog, dict):
            runtime_groups: Dict[tuple, List[str]] = {}
            meta_groups: Dict[str, List[str]] = {}
            for group, items in source_catalog.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    operation = item.get("operation")
                    if not operation:
                        continue
                    runtime_key = (
                        item.get("runtime_family") or "",
                        item.get("data_contract") or "",
                    )
                    runtime_groups.setdefault(runtime_key, []).append(operation)
                    meta = item.get("fedot_industrial_meta") or ""
                    if meta:
                        meta_groups.setdefault(str(meta), []).append(operation)
            cleaned["runtime_contracts"] = [
                {
                    "runtime_family": runtime_family,
                    "data_contract": data_contract,
                    "operations": operations_for_contract,
                }
                for (runtime_family, data_contract), operations_for_contract in runtime_groups.items()
            ]
            cleaned["fedot_industrial_meta_groups"] = meta_groups
            adapters = []
            for group, items in source_catalog.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict) or not item.get("runtime_adapter"):
                        continue
                    adapters.append({
                        "operation": item.get("operation"),
                        "adapter": item.get("runtime_adapter"),
                        "train_only": bool(item.get("train_only")),
                        "requires_categorical_metadata": bool(item.get("requires_categorical_metadata")),
                    })
            if adapters:
                cleaned["runtime_adapters"] = adapters
        if isinstance(operations.get("default_graph"), list):
            cleaned["default_graph"] = [
                {key: value for key, value in node.items() if key != "params"}
                for node in operations["default_graph"]
                if isinstance(node, dict)
            ]
        if isinstance(operations.get("industrial_templates"), list):
            templates = []
            for template in operations["industrial_templates"]:
                if not isinstance(template, dict):
                    continue
                item = {
                    "name": template.get("name"),
                    "description": template.get("description"),
                    "graph": template.get("graph"),
                }
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
        strategies = operations.get("industrial_strategies_catalog")
        if isinstance(strategies, list):
            cleaned["industrial_strategies_catalog"] = [
                {
                    "name": item.get("name"),
                    "label": item.get("label"),
                    "default_params": {
                        key: value
                        for key, value in dict(item.get("default_params", {}) or {}).items()
                        if key in {"data_type", "problem", "available_operations"}
                    },
                }
                for item in strategies
                if isinstance(item, dict)
            ]
        return cleaned

    @staticmethod
    def _compact_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(profile, dict):
            return {}
        compact = dict(profile)
        for key in ("feature_names", "numeric_feature_names", "categorical_feature_names"):
            if isinstance(compact.get(key), list):
                compact[key] = compact[key][:12]
        stats = dict(compact.get("feature_stats", {}) or {})
        card = stats.get("categorical_cardinalities")
        if isinstance(card, dict):
            stats["categorical_cardinalities"] = dict(list(card.items())[:10])
        compact["feature_stats"] = stats
        return compact
