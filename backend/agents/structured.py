"""Pydantic schemas for LLM-emitted structured objects."""

from __future__ import annotations

import json
from typing import Any, Dict, List

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

    def as_graph_json(self) -> str:
        return self.json()


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
        candidate = {"task_type": args.get("task_type"), "nodes": args.get("nodes")}
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
