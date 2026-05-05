"""Architect: asks the LLM to synthesize validated pipeline graphs."""

import json
import logging
from typing import Optional

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
      {"id": "model", "operation": "operation_from_catalog", "params": {}, "inputs": []}
    ]
  },
  "analysis": "why this graph fits the evidence",
  "reasoning": "why these operations/params were selected from the catalog"
}

Use only operations listed in available_operations. Every input must be an
existing node id, or [] for raw data. The graph must be a DAG with exactly one
root model node. The graph passed to Fedot.Industrial as `initial_assumption` is
finetuned by AutoML. Industrial strategies (federated_automl, sampling_strategy)
are execution modes selected outside the graph; do not include them in the JSON.
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
            "industrial_strategy_in_use": dc.industrial_strategy or "tabular",
            "industrial_strategy_params": dict(dc.industrial_strategy_params or {}),
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
            "The graph you produce becomes Fedot.Industrial `initial_assumption` and is polished by AutoML finetune.\n"
            "available_operations.catalog[].industrial_search_space contains Fedot.Industrial-supported parameter ranges.\n"
            "available_operations.industrial_templates contains Fedot.Industrial-native graph patterns.\n"
            "available_operations.industrial_strategies_catalog describes selectable execution strategies "
            "(tabular/federated_automl/sampling_strategy). Strategies are picked by the user outside the graph; "
            "do not include them in your JSON.\n"
            "For graph node inputs, use [] for raw data and otherwise use only existing node ids.\n"
            "Return strict JSON literals: null/true/false, not Python None/True/False.\n"
            "Return only the GraphProposal JSON object.\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
