from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .base import ToolSpec, ToolRunner, ExecutionResult, registry


@dataclass
class SearchTool(ToolRunner):
    def __init__(self):
        super().__init__(
            ToolSpec(
                name="search",
                description="Retrieve snippets for a query (placeholder).",
                inputs={"query": "TEXT"},
                outputs={"snippets": "TEXT"},
            )
        )

    def run(self, params: Dict[str, str]) -> ExecutionResult:
        ok, err = self.validate(params)
        if not ok:
            return ExecutionResult(success=False, error=err)
        q = params["query"].strip()
        if not q:
            return ExecutionResult(success=False, error="Empty query")
        return ExecutionResult(
            success=True, data={"snippets": f"Results for: {q}"}, metrics={"hits": 3}
        )


@dataclass
class SummarizeTool(ToolRunner):
    def __init__(self):
        super().__init__(
            ToolSpec(
                name="summarize",
                description="Summarize provided snippets (placeholder).",
                inputs={"snippets": "TEXT"},
                outputs={"summary": "TEXT"},
            )
        )

    def run(self, params: Dict[str, str]) -> ExecutionResult:
        ok, err = self.validate(params)
        if not ok:
            return ExecutionResult(success=False, error=err)
        s = params["snippets"].strip()
        if not s:
            return ExecutionResult(success=False, error="Empty snippets")
        return ExecutionResult(success=True, data={"summary": f"Summary: {s[:60]}..."})


registry.register("search", SearchTool)
registry.register("summarize", SummarizeTool)
