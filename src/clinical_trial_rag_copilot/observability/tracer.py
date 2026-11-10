import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

# Try importing LangSmith traceable decorator
try:
    from langsmith import traceable
except ImportError:
    # Fallback no-op decorator if langsmith is not installed
    def traceable(name: Optional[str] = None, run_type: str = "chain", **kwargs):
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

TRACES_DIR = Path(__file__).resolve().parent.parent / "data" / "output" / "traces"
TRACES_DIR.mkdir(parents=True, exist_ok=True)


class TraceSpan(BaseModel):
    """Structured execution span capturing telemetry for a pipeline step."""
    span_id: str
    step_name: str
    start_time: str
    end_time: str
    duration_ms: float
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    status: str = "SUCCESS"  # "SUCCESS" or "ERROR"
    error_message: Optional[str] = None


class PipelineTrace(BaseModel):
    """Complete trace of an end-to-end GraphRAG query execution."""
    trace_id: str
    query: str
    timestamp: str
    total_latency_ms: float
    guardrail_status: str
    spans: List[TraceSpan] = Field(default_factory=list)


class PipelineTracer:
    """Production-grade telemetry tracer recording spans locally and to LangSmith."""

    def __init__(self, trace_id: Optional[str] = None):
        import uuid
        self.trace_id = trace_id or f"trace_{uuid.uuid4().hex[:12]}"
        self.spans: List[TraceSpan] = []
        self.start_timestamp = time.time()
        self.iso_timestamp = datetime.now(timezone.utc).isoformat()

    def start_span(self, step_name: str, inputs: Optional[Dict[str, Any]] = None) -> "_SpanContext":
        """Context manager to record a pipeline execution span."""
        return _SpanContext(self, step_name, inputs or {})

    def add_span(self, span: TraceSpan):
        self.spans.append(span)

    def finalize(self, query: str, guardrail_status: str) -> PipelineTrace:
        """Finalize the trace and persist to disk."""
        total_latency = (time.time() - self.start_timestamp) * 1000
        trace = PipelineTrace(
            trace_id=self.trace_id,
            query=query,
            timestamp=self.iso_timestamp,
            total_latency_ms=round(total_latency, 2),
            guardrail_status=guardrail_status,
            spans=self.spans,
        )

        # Save to local traces log
        trace_file = TRACES_DIR / f"{self.trace_id}.json"
        try:
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump(trace.model_dump(), f, indent=2)
        except Exception as e:
            print(f"[Tracer] Warning: Could not save trace to disk: {e}")

        return trace


class _SpanContext:
    """Internal context manager for span timing."""

    def __init__(self, tracer: PipelineTracer, step_name: str, inputs: Dict[str, Any]):
        self.tracer = tracer
        self.step_name = step_name
        self.inputs = inputs
        self.start_time = 0.0
        self.start_iso = ""

    def __enter__(self):
        self.start_time = time.time()
        self.start_iso = datetime.now(timezone.utc).isoformat()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.time() - self.start_time) * 1000
        end_iso = datetime.now(timezone.utc).isoformat()

        status = "ERROR" if exc_type else "SUCCESS"
        err_msg = str(exc_val) if exc_val else None

        span = TraceSpan(
            span_id=f"span_{len(self.tracer.spans)+1}",
            step_name=self.step_name,
            start_time=self.start_iso,
            end_time=end_iso,
            duration_ms=round(duration, 2),
            inputs=self._sanitize(self.inputs),
            outputs={},
            status=status,
            error_message=err_msg,
        )
        self.tracer.add_span(span)
        return False

    def set_outputs(self, outputs: Dict[str, Any]):
        """Attach outputs to the latest span."""
        if self.tracer.spans:
            self.tracer.spans[-1].outputs = self._sanitize(outputs)

    def _sanitize(self, data: Any) -> Any:
        """Ensure objects are JSON serializable."""
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, dict):
            return {str(k): self._sanitize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize(v) for v in data[:20]]  # Cap long lists
        elif hasattr(data, "model_dump"):
            return data.model_dump()
        else:
            return str(data)[:200]


def get_recent_traces(limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch the most recent execution traces."""
    traces = []
    if not TRACES_DIR.exists():
        return []

    files = sorted(TRACES_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    for f in files[:limit]:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                traces.append(json.load(fp))
        except Exception:
            continue
    return traces
