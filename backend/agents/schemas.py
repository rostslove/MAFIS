"""Shared dataclasses for inter-agent communication."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ToolCall:
    """Record of a single MCP tool call."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "args": self.arguments,
            "result": str(self.result)[:500],
            "success": self.success,
            "error": self.error,
        }


@dataclass
class IterationRecord:
    """Compact record of one approved graph evaluation for agent memory."""
    iteration: int = 0
    graph: Dict[str, Any] = field(default_factory=dict)
    graph_score: float = 0.0
    winner: str = ""
    suggested_mutations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DataContext:
    """Shared context passed between agents (no raw arrays - tools work on csv_path)."""
    csv_path: str
    target_column: str
    task_type: str
    profile: Dict[str, Any] = field(default_factory=dict)
    forecast_length: Optional[int] = None
    primary_metric: Optional[str] = None
    test_size: float = 0.2
    iteration_history: List[IterationRecord] = field(default_factory=list)

    @property
    def is_time_series(self) -> bool:
        return self.task_type in ("ts_classification", "ts_regression", "ts_forecasting")


@dataclass
class ArchitectResult:
    graph: Dict[str, Any] = field(default_factory=dict)  # PipelineGraph as dict
    mermaid: str = ""
    analysis: str = ""
    reasoning: str = ""
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph": self.graph,
            "mermaid": self.mermaid,
            "analysis": self.analysis,
            "reasoning": self.reasoning,
            "diagnostics": self.diagnostics,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }


@dataclass
class EngineerResult:
    graph_score: float = 0.0
    graph_metrics: Dict[str, Any] = field(default_factory=dict)
    train_metrics: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    split_info: Dict[str, Any] = field(default_factory=dict)
    tuned_nodes: List[Dict[str, Any]] = field(default_factory=list)
    graph_error: str = ""
    target_info: Dict[str, Any] = field(default_factory=dict)
    training_notes: List[str] = field(default_factory=list)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_score": self.graph_score,
            "graph_metrics": self.graph_metrics,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "split_info": self.split_info,
            "tuned_nodes": self.tuned_nodes,
            "graph_error": self.graph_error,
            "target_info": self.target_info,
            "training_notes": self.training_notes,
            "diagnostics": self.diagnostics,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }


@dataclass
class CriticFeedback:
    winner: str = ""  # "accepted" or "needs_revision"
    assessment: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggested_mutations: List[Dict[str, Any]] = field(default_factory=list)
    improvement_plan: List[str] = field(default_factory=list)
    should_stop: bool = False
    node_importance: Dict[str, float] = field(default_factory=dict)
    explanation: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    full_response: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner,
            "assessment": self.assessment,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggested_mutations": self.suggested_mutations,
            "improvement_plan": self.improvement_plan,
            "should_stop": self.should_stop,
            "node_importance": self.node_importance,
            "explanation": self.explanation,
            "diagnostics": self.diagnostics,
            "full_response": self.full_response,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }


@dataclass
class ScribeReport:
    title: str = ""
    summary: str = ""
    methodology: str = ""
    results: str = ""
    recommendations: List[str] = field(default_factory=list)
    best_graph_mermaid: str = ""
    full_response: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "methodology": self.methodology,
            "results": self.results,
            "recommendations": self.recommendations,
            "best_graph_mermaid": self.best_graph_mermaid,
            "full_response": self.full_response,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }
