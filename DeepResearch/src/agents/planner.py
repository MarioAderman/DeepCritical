from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Planner:
    """Placeholder planner that mirrors Parser/PlannerAgent logic with rewrite/search/finalize."""

    def plan(self, question: str) -> List[Dict[str, Any]]:
        return [
            {"tool": "rewrite", "params": {"query": question}},
            {"tool": "web_search", "params": {"query": "${rewrite.queries}"}},
            {"tool": "summarize", "params": {"snippets": "${web_search.results}"}},
            {
                "tool": "references",
                "params": {
                    "answer": "${summarize.summary}",
                    "web": "${web_search.results}",
                },
            },
            {"tool": "finalize", "params": {"draft": "${references.answer_with_refs}"}},
            {
                "tool": "evaluator",
                "params": {"question": question, "answer": "${finalize.final}"},
            },
        ]
