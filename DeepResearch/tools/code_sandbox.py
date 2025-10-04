from __future__ import annotations

import json
import re
from dataclasses import dataclass
from textwrap import indent
from typing import Any, Dict, List

from .base import ToolSpec, ToolRunner, ExecutionResult, registry


SAFE_BUILTINS: Dict[str, Any] = {
    # Whitelist of safe Python builtins for sandboxed execution
    "abs": abs,
    "all": all,
    "any": any,
    "enumerate": enumerate,
    "filter": filter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "sorted": sorted,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        cleaned = re.sub(r"\s+", " ", value.replace("\n", " ")).strip()
        return f'"{cleaned[:47]}..."' if len(cleaned) > 50 else f'"{cleaned}"'
    if isinstance(value, (int, float, bool)):
        return str(value)
    if hasattr(value, "isoformat"):
        try:
            return f'"{value.isoformat()}"'
        except Exception:
            return ""  # fallback
    return ""


def _analyze_structure(value: Any, indent_str: str = "") -> str:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return f"{type(value).__name__}{f' (example: {_format_value(value)})' if _format_value(value) else ''}"
    if isinstance(value, list):
        if not value:
            return "Array<unknown>"
        return f"Array<{_analyze_structure(value[0], indent_str + '  ')}>"
    if isinstance(value, dict):
        if not value:
            return "{}"
        props: List[str] = []
        for k, v in value.items():
            analyzed = _analyze_structure(v, indent_str + "  ")
            props.append(f'{indent_str}  "{k}": {analyzed}')
        return "{\n" + ",\n".join(props) + f"\n{indent_str}" + "}"
    # Fallback
    return type(value).__name__


def _dict_from_context(context_str: str) -> Dict[str, Any]:
    if not context_str:
        return {}
    try:
        ctx = json.loads(context_str)
        return ctx if isinstance(ctx, dict) else {}
    except Exception:
        return {}


def _extract_code_from_output(text: str) -> str:
    # Try to extract fenced code block first
    fence = re.search(r"```[a-zA-Z0-9_]*\n([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()
    return text.strip()


@dataclass
class CodeSandboxRunner(ToolRunner):
    def __init__(self):
        super().__init__(
            ToolSpec(
                name="code_sandbox",
                description="Generate and evaluate Python code for a given problem within a sandbox.",
                inputs={"problem": "TEXT", "context": "TEXT", "max_attempts": "TEXT"},
                outputs={"code": "TEXT", "output": "TEXT"},
            )
        )

    def _generate_code(
        self, problem: str, available_vars: str, previous_attempts: List[Dict[str, str]]
    ) -> str:
        # Load prompt from Hydra via PromptLoader; fall back to a minimal system
        try:
            from DeepResearch.src.prompts import PromptLoader  # type: ignore

            cfg: Dict[str, Any] = {}
            loader = PromptLoader(cfg)  # type: ignore
            system = loader.get("code_sandbox")
        except Exception:
            system = (
                "You are an expert Python programmer. Generate Python code that returns the result directly.\n"
                "Available variables (one can be used directly as identifiers):\n"
                f"{available_vars}\nMust include a return statement."
            )

        previous_ctx = "\n".join(
            [
                f"<bad-attempt-{i + 1}>\n{a.get('code', '')}\nError: {a.get('error', '')}\n</bad-attempt-{i + 1}>"
                for i, a in enumerate(previous_attempts)
            ]
        )

        previous_section = (
            ("Previous attempts and their errors:\n" + previous_ctx)
            if previous_attempts
            else ""
        )
        user_prompt = (
            f"Problem: {problem}\n\n"
            f"Available variables:\n{available_vars}\n\n"
            f"{previous_section}"
            "Respond with ONLY the code body without explanations."
        )

        # Use pydantic_ai Agent like other runners
        try:
            from DeepResearch.tools.pyd_ai_tools import _build_agent  # type: ignore

            agent, _ = _build_agent({}, [], [])
            if agent is None:
                raise RuntimeError("pydantic_ai not available")
            result = agent.run_sync({"instructions": system, "input": user_prompt})
            output_text = getattr(result, "output", str(result))
        except Exception:
            # Fallback: minimal template to ensure progress
            output_text = "return None"

        return _extract_code_from_output(output_text)

    def _evaluate_code(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Prepare locals with context variables (valid identifiers only)
        locals_env: Dict[str, Any] = {}
        for key, value in (context or {}).items():
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                locals_env[key] = value

        # Wrap code into a function to capture return value
        wrapped = (
            f"def __solution__():\n{indent(code, '    ')}\nresult = __solution__()"
        )
        global_env: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS}

        try:
            exec(wrapped, global_env, locals_env)
        except Exception as e:
            return {"success": False, "error": str(e)}

        if "result" not in locals_env:
            return {
                "success": False,
                "error": "No value was returned, make sure to use 'return' statement to return the result",
            }
        return {"success": True, "output": locals_env["result"]}

    def run(self, params: Dict[str, str]) -> ExecutionResult:
        ok, err = self.validate(params)
        if not ok:
            return ExecutionResult(success=False, error=err)

        problem = params.get("problem", "").strip()
        context_str = params.get("context", "").strip()
        max_attempts_str = params.get("max_attempts", "3").strip()

        if not problem:
            return ExecutionResult(success=False, error="Empty problem")

        try:
            max_attempts = max(1, int(max_attempts_str))
        except Exception:
            max_attempts = 3

        ctx = _dict_from_context(context_str)
        available_vars = _analyze_structure(ctx)

        attempts: List[Dict[str, str]] = []

        for _ in range(max_attempts):
            code = self._generate_code(problem, available_vars, attempts)
            eval_result = self._evaluate_code(code, ctx)
            if eval_result.get("success"):
                return ExecutionResult(
                    success=True,
                    data={
                        "code": code,
                        "output": str(eval_result.get("output")),
                    },
                )
            attempts.append(
                {"code": code, "error": str(eval_result.get("error", "Unknown error"))}
            )

        return ExecutionResult(
            success=False,
            error=f"Failed to generate working code after {max_attempts} attempts",
        )


# Register tool
registry.register("code_sandbox", CodeSandboxRunner)
