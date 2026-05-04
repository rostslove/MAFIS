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
    ],
    "training_strategy": null
  },
  "analysis": "why this graph fits the evidence",
  "reasoning": "why these operations/params were selected from the catalog"
}

Use only operations listed in available_operations. Every input must be an
existing node id, or [] for raw data. The graph must be a DAG with exactly one
root model node. Use training_strategy only when the payload policy allows it.
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
            self._enforce_training_strategy_policy(proposal, prev_feedback, prev_graph, diagnostics)

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

    def _enforce_training_strategy_policy(self, proposal, prev_feedback, prev_graph, diagnostics) -> None:
        if not proposal.graph.training_strategy or self._training_strategy_allowed(prev_feedback, prev_graph):
            return
        proposal.graph.training_strategy = None
        diagnostics.append(
            {
                "agent": "Architect",
                "kind": "training_strategy_removed",
                "summary": "Architect proposed a training strategy without an explicit request or Critic recommendation.",
                "technical_message": (
                    "training_strategy was reset to null. Strategies are execution modes and "
                    "must be selected by the user, preserved from the previous graph, or recommended by Critic."
                ),
                "recommendations": [
                    "Keep graph.training_strategy null unless strategy use is explicitly requested.",
                    "Use Critic set_strategy feedback when a Fedot.Industrial strategy should be selected.",
                ],
                "recoverable": True,
            }
        )

    def _build_structured_prompt(self, dc: DataContext, profile, operations, prev_fb, prev_graph) -> str:
        strategy_allowed = self._training_strategy_allowed(prev_fb, prev_graph)
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
            "training_strategy_policy": {
                "default": "null",
                "allowed_now": strategy_allowed,
                "rule": (
                    "Use a training strategy only if previous_graph already has one "
                    "or Critic suggested a set_strategy mutation."
                ),
                "schema": {"name": "strategy_name", "params": {}},
            },
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
            "available_operations.catalog[].industrial_search_space contains Fedot.Industrial-supported parameter ranges.\n"
            "available_operations.industrial_templates contains Fedot.Industrial-native graph patterns.\n"
            "available_operations.training_strategies_catalog contains selectable execution strategies.\n"
            "For graph node inputs, use [] for raw data and otherwise use only existing node ids.\n"
            "Return strict JSON literals: null/true/false, not Python None/True/False.\n"
            "Return only the GraphProposal JSON object.\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    @staticmethod
    def _training_strategy_allowed(prev_fb, prev_graph) -> bool:
        if isinstance(prev_graph, dict) and prev_graph.get("training_strategy"):
            return True
        if not prev_fb:
            return False
        mutations = getattr(prev_fb, "suggested_mutations", []) or []
        return any(
            (mutation or {}).get("type") == "set_strategy"
            for mutation in mutations
            if isinstance(mutation, dict)
        )
