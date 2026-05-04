"""Pydantic schemas for LLM-emitted structured objects."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field, ValidationError, root_validator, validator
except ImportError as exc:  # pragma: no cover - backend env should include pydantic via Fedot
    raise RuntimeError("pydantic is required for structured LLM object validation") from exc


class GraphNodeObject(BaseModel):
    id: str
    operation: str
    params: Dict[str, Any] = Field(default_factory=dict)
    inputs: List[str] = Field(default_factory=list)

    @validator("id", "operation")
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must be non-empty")
        return value


class GraphObject(BaseModel):
    task_type: str
    nodes: List[GraphNodeObject]
    training_strategy: Optional[Dict[str, Any]] = None

    @validator("task_type")
    def non_empty_task(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("task_type must be non-empty")
        return value

    @validator("nodes")
    def non_empty_nodes(cls, value: List[GraphNodeObject]) -> List[GraphNodeObject]:
        if not value:
            raise ValueError("graph must contain at least one node")
        return value

    @validator("training_strategy", pre=True, always=True)
    def normalize_training_strategy(cls, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if isinstance(value, str):
            name = value.strip()
            if not name or name.lower() in {"none", "null", "default", "direct graph", "graph"}:
                return None
            return {"name": name, "params": {}}
        if isinstance(value, dict):
            return value
        raise ValueError("training_strategy must be null, a strategy name, or an object")

    def as_graph_json(self) -> str:
        return self.json()


class GraphProposalObject(BaseModel):
    graph: GraphObject
    analysis: str = ""
    reasoning: str = ""

    @validator("analysis", "reasoning", pre=True, always=True)
    def stringify_text(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()


class CriticMutationObject(BaseModel):
    type: str
    node_id: Optional[str] = None
    new_operation: Optional[str] = None
    node: Optional[GraphNodeObject] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    strategy: Optional[Dict[str, Any]] = None
    rewire_input_of: Optional[str] = None
    input_id: Optional[str] = None

    @validator("type")
    def allowed_type(cls, value: str) -> str:
        value = value.strip()
        allowed = {"remove", "replace", "add", "set_params", "set_strategy", "clear_strategy", "connect"}
        if value not in allowed:
            raise ValueError(f"unsupported mutation type: {value}")
        return value

    @validator("node_id", "new_operation", "rewire_input_of", "input_id", pre=True, always=True)
    def optional_string(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def as_mutation_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": self.type}
        if self.node_id:
            payload["node_id"] = self.node_id
        if self.new_operation:
            payload["new_operation"] = self.new_operation
        if self.node is not None:
            payload["node"] = self.node.dict()
        if self.params:
            payload["params"] = self.params
        if self.strategy:
            payload["strategy"] = self.strategy
        if self.rewire_input_of:
            payload["rewire_input_of"] = self.rewire_input_of
        if self.input_id:
            payload["input_id"] = self.input_id
        return payload


class CriticFeedbackObject(BaseModel):
    assessment: str = ""
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    suggested_mutations: List[CriticMutationObject] = Field(default_factory=list)
    improvement_plan: List[str] = Field(default_factory=list)
    should_stop: bool = False

    @validator("assessment", pre=True, always=True)
    def stringify_assessment(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()

    @validator("strengths", "weaknesses", "improvement_plan", pre=True, always=True)
    def string_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("must be a list of strings")
        return [str(item).strip() for item in value if str(item).strip()]

    @validator("suggested_mutations", pre=True, always=True)
    def mutation_list(cls, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        if not isinstance(value, list):
            raise ValueError("suggested_mutations must be a list")
        return [item for item in value if isinstance(item, dict)]

    def mutations_as_dicts(self) -> List[Dict[str, Any]]:
        return [mutation.as_mutation_dict() for mutation in self.suggested_mutations]


class ScribeReportObject(BaseModel):
    title: str = ""
    summary: str = ""
    methodology: str = ""
    results: str = ""
    recommendations: List[str] = Field(default_factory=list)

    @validator("title", "summary", "methodology", "results", pre=True, always=True)
    def stringify_text(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()

    @validator("recommendations", pre=True, always=True)
    def recommendation_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("recommendations must be a list")
        return [str(item).strip() for item in value if str(item).strip()]


class ToolCallObject(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

    @root_validator(pre=True)
    def accept_aliases_and_parse_arguments(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(values, dict):
            raise ValueError("tool call must be an object")

        name = values.get("name") or values.get("tool") or values.get("tool_name")
        args = values.get("arguments") if "arguments" in values else values.get("args", {})

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                raise ValueError("arguments must be a JSON object, not malformed JSON text") from exc
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")

        if name == "propose_graph":
            args = normalize_propose_graph_arguments(args)
        elif name == "mutate_graph":
            args = normalize_mutate_graph_arguments(args)

        return {"name": name, "arguments": args}

    @validator("name")
    def non_empty_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("tool name must be non-empty")
        return value


def normalize_propose_graph_arguments(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_json = args.get("graph_json")
    if isinstance(graph_json, dict):
        args["graph_json"] = GraphObject.parse_obj(graph_json).as_graph_json()
    elif isinstance(graph_json, str):
        args["graph_json"] = GraphObject.parse_raw(graph_json).as_graph_json()
    else:
        candidate = {
            "task_type": args.get("task_type"),
            "nodes": args.get("nodes"),
            "training_strategy": args.get("training_strategy", args.get("strategy")),
        }
        args["graph_json"] = GraphObject.parse_obj(candidate).as_graph_json()
    return {"graph_json": args["graph_json"]}


def normalize_mutate_graph_arguments(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_json = args.get("graph_json")
    mutation_json = args.get("mutation_json")
    if isinstance(graph_json, dict):
        graph_json = GraphObject.parse_obj(graph_json).as_graph_json()
    elif isinstance(graph_json, str):
        graph_json = GraphObject.parse_raw(graph_json).as_graph_json()
    else:
        raise ValueError("mutate_graph requires graph_json")

    if isinstance(mutation_json, dict):
        mutation_json = json.dumps(mutation_json)
    elif isinstance(mutation_json, str):
        parsed = json.loads(mutation_json)
        if not isinstance(parsed, dict):
            raise ValueError("mutation_json must decode to an object")
        mutation_json = json.dumps(parsed)
    else:
        raise ValueError("mutate_graph requires mutation_json")
    return {"graph_json": graph_json, "mutation_json": mutation_json}


def parse_tool_call_object(payload: Any) -> ToolCallObject | None:
    try:
        return ToolCallObject.parse_obj(payload)
    except ValidationError:
        return None


def parse_graph_proposal_object(payload: Any) -> GraphProposalObject | None:
    try:
        return GraphProposalObject.parse_obj(payload)
    except ValidationError:
        return None


def parse_critic_feedback_object(payload: Any) -> CriticFeedbackObject | None:
    try:
        return CriticFeedbackObject.parse_obj(payload)
    except ValidationError:
        return None


def parse_scribe_report_object(payload: Any) -> ScribeReportObject | None:
    try:
        return ScribeReportObject.parse_obj(payload)
    except ValidationError:
        return None
