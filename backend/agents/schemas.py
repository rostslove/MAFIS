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
        }


@dataclass
class IterationRecord:
    """Compact record of one iteration for cross-iteration memory."""
    iteration: int = 0
    graph: Dict[str, Any] = field(default_factory=dict)
    graph_score: float = 0.0
    best_baseline_score: float = 0.0
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
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph": self.graph,
            "mermaid": self.mermaid,
            "analysis": self.analysis,
            "reasoning": self.reasoning,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }


@dataclass
class BaselineResult:
    name: str
    score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class EngineerResult:
    baseline_results: List[BaselineResult] = field(default_factory=list)
    graph_score: float = 0.0
    graph_metrics: Dict[str, float] = field(default_factory=dict)
    tuned_nodes: List[Dict[str, Any]] = field(default_factory=list)
    best_baseline_score: float = 0.0
    best_baseline_name: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_results": [
                {"name": r.name, "score": r.score, "metrics": r.metrics, "error": r.error}
                for r in self.baseline_results
            ],
            "graph_score": self.graph_score,
            "graph_metrics": self.graph_metrics,
            "tuned_nodes": self.tuned_nodes,
            "best_baseline_score": self.best_baseline_score,
            "best_baseline_name": self.best_baseline_name,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }


@dataclass
class CriticFeedback:
    winner: str = ""  # "graph" or "baseline"
    assessment: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggested_mutations: List[Dict[str, Any]] = field(default_factory=list)
    should_stop: bool = False
    node_importance: Dict[str, float] = field(default_factory=dict)
    explanation: Dict[str, Any] = field(default_factory=dict)
    full_response: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner,
            "assessment": self.assessment,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggested_mutations": self.suggested_mutations,
            "should_stop": self.should_stop,
            "node_importance": self.node_importance,
            "explanation": self.explanation,
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
