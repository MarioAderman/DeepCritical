from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from time import sleep
from typing import Any, Dict, Optional, ClassVar

from .base import ToolSpec, ToolRunner, ExecutionResult, registry

# Configure logging
logger = logging.getLogger(__name__)

# Timeout message for when execution times out
TIMEOUT_MSG = "Execution timed out after the specified timeout period."


def _get_cfg_value(cfg: Dict[str, Any], path: str, default: Any) -> Any:
    """Get nested configuration value using dot notation."""
    cur: Any = cfg
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def _get_file_name_from_content(code: str, work_dir: Path) -> Optional[str]:
    """Extract filename from code content comments, similar to AutoGen implementation."""
    lines = code.split("\n")
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if line.startswith("# filename:") or line.startswith("# file:"):
            filename = line.split(":", 1)[1].strip()
            # Basic validation - ensure it's a valid filename
            if filename and not os.path.isabs(filename) and ".." not in filename:
                return filename
    return None


def _cmd(language: str) -> str:
    """Get the command to execute code for a given language."""
    language = language.lower()
    if language == "python":
        return "python"
    elif language in ["bash", "shell", "sh"]:
        return "sh"
    elif language in ["pwsh", "powershell", "ps1"]:
        return "pwsh"
    else:
        return language


def _wait_for_ready(container, timeout: int = 60, stop_time: float = 0.1) -> None:
    """Wait for container to be ready, similar to AutoGen implementation."""
    elapsed_time = 0.0
    while container.status != "running" and elapsed_time < timeout:
        sleep(stop_time)
        elapsed_time += stop_time
        container.reload()
        continue
    if container.status != "running":
        raise ValueError("Container failed to start")


@dataclass
class DockerSandboxRunner(ToolRunner):
    """Enhanced Docker sandbox runner using Testcontainers with AutoGen-inspired patterns."""

    # Default execution policies similar to AutoGen
    DEFAULT_EXECUTION_POLICY: ClassVar[Dict[str, bool]] = {
        "bash": True,
        "shell": True,
        "sh": True,
        "pwsh": True,
        "powershell": True,
        "ps1": True,
        "python": True,
        "javascript": False,
        "html": False,
        "css": False,
    }

    # Language aliases
    LANGUAGE_ALIASES: ClassVar[Dict[str, str]] = {"py": "python", "js": "javascript"}

    def __init__(self):
        super().__init__(
            ToolSpec(
                name="docker_sandbox",
                description="Run code/command in an isolated container using Testcontainers with enhanced execution policies.",
                inputs={
                    "language": "TEXT",  # e.g., python, bash, shell, sh, pwsh, powershell, ps1
                    "code": "TEXT",  # code string to execute
                    "command": "TEXT",  # explicit command to run (overrides code when provided)
                    "env": "TEXT",  # JSON of env vars
                    "timeout": "TEXT",  # seconds
                    "execution_policy": "TEXT",  # JSON dict of language->bool execution policies
                },
                outputs={
                    "stdout": "TEXT",
                    "stderr": "TEXT",
                    "exit_code": "TEXT",
                    "files": "TEXT",
                },
            )
        )

        # Initialize execution policies
        self.execution_policies = self.DEFAULT_EXECUTION_POLICY.copy()

    def run(self, params: Dict[str, Any]) -> ExecutionResult:
        """Execute code in a Docker container with enhanced error handling and execution policies."""
        ok, err = self.validate(params)
        if not ok:
            return ExecutionResult(success=False, error=err)

        # Parse parameters
        language = str(params.get("language", "python")).strip() or "python"
        code = str(params.get("code", "")).strip()
        explicit_cmd = str(params.get("command", "")).strip()
        env_json = str(params.get("env", "")).strip()
        timeout_str = str(params.get("timeout", "60")).strip()
        execution_policy_json = str(params.get("execution_policy", "")).strip()

        # Parse timeout
        try:
            timeout = max(1, int(timeout_str))
        except Exception:
            timeout = 60

        # Parse environment variables
        try:
            env_map: Dict[str, str] = json.loads(env_json) if env_json else {}
            if not isinstance(env_map, dict):
                env_map = {}
        except Exception:
            env_map = {}

        # Parse execution policies
        try:
            if execution_policy_json:
                custom_policies = json.loads(execution_policy_json)
                if isinstance(custom_policies, dict):
                    self.execution_policies.update(custom_policies)
        except Exception:
            pass  # Use default policies

        # Load hydra config if accessible to configure container image and limits
        try:
            cfg: Dict[str, Any] = {}
        except Exception:
            cfg = {}

        # Get configuration values
        image = _get_cfg_value(cfg, "sandbox.image", "python:3.11-slim")
        workdir = _get_cfg_value(cfg, "sandbox.workdir", "/workspace")
        cpu = _get_cfg_value(cfg, "sandbox.cpu", None)
        mem = _get_cfg_value(cfg, "sandbox.mem", None)
        auto_remove = _get_cfg_value(cfg, "sandbox.auto_remove", True)

        # Normalize language and check execution policy
        lang = self.LANGUAGE_ALIASES.get(language.lower(), language.lower())
        if lang not in self.DEFAULT_EXECUTION_POLICY:
            return ExecutionResult(success=False, error=f"Unsupported language: {lang}")

        execute_code = self.execution_policies.get(lang, False)
        if not execute_code and not explicit_cmd:
            return ExecutionResult(
                success=False, error=f"Execution disabled for language: {lang}"
            )

        try:
            from testcontainers.core.container import DockerContainer
        except Exception as e:
            return ExecutionResult(
                success=False, error=f"testcontainers unavailable: {e}"
            )

        # Prepare working directory
        temp_dir: Optional[str] = None
        work_path = Path(tempfile.mkdtemp(prefix="docker-sandbox-"))
        files_created = []

        try:
            # Create container with enhanced configuration
            container_name = f"deepcritical-sandbox-{uuid.uuid4().hex[:8]}"
            container = DockerContainer(image)
            container.with_name(container_name)

            # Set environment variables
            container.with_env("PYTHONUNBUFFERED", "1")
            for k, v in (env_map or {}).items():
                container.with_env(str(k), str(v))

            # Set resource limits if configured
            if cpu:
                try:
                    container.with_cpu_quota(int(cpu))
                except Exception:
                    logger.warning(f"Failed to set CPU quota: {cpu}")

            if mem:
                try:
                    container.with_memory(mem)
                except Exception:
                    logger.warning(f"Failed to set memory limit: {mem}")

            container.with_workdir(workdir)

            # Mount working directory
            container.with_volume_mapping(str(work_path), workdir)

            # Handle code execution
            if explicit_cmd:
                # Use explicit command
                cmd = explicit_cmd
                container.with_command(cmd)
            else:
                # Save code to file and execute
                filename = _get_file_name_from_content(code, work_path)
                if not filename:
                    filename = f"tmp_code_{md5(code.encode()).hexdigest()}.{lang}"

                code_path = work_path / filename
                with code_path.open("w", encoding="utf-8") as f:
                    f.write(code)
                files_created.append(str(code_path))

                # Build execution command
                if lang == "python":
                    cmd = ["python", filename]
                elif lang in ["bash", "shell", "sh"]:
                    cmd = ["sh", filename]
                elif lang in ["pwsh", "powershell", "ps1"]:
                    cmd = ["pwsh", filename]
                else:
                    cmd = [_cmd(lang), filename]

                container.with_command(cmd)

            # Start container and wait for readiness
            logger.info(f"Starting container {container_name} with image {image}")
            container.start()
            _wait_for_ready(container, timeout=30)

            # Execute the command with timeout
            logger.info(f"Executing command: {cmd}")
            result = container.get_wrapped_container().exec_run(
                cmd,
                workdir=workdir,
                environment=env_map,
                stdout=True,
                stderr=True,
                demux=True,
            )

            # Parse results
            stdout_bytes, stderr_bytes = (
                result.output
                if isinstance(result.output, tuple)
                else (result.output, b"")
            )
            exit_code = result.exit_code

            # Decode output
            stdout = (
                stdout_bytes.decode("utf-8", errors="replace")
                if isinstance(stdout_bytes, (bytes, bytearray))
                else str(stdout_bytes)
            )
            stderr = (
                stderr_bytes.decode("utf-8", errors="replace")
                if isinstance(stderr_bytes, (bytes, bytearray))
                else ""
            )

            # Handle timeout
            if exit_code == 124:
                stderr += "\n" + TIMEOUT_MSG

            # Stop container
            container.stop()

            return ExecutionResult(
                success=True,
                data={
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": str(exit_code),
                    "files": json.dumps(files_created),
                },
            )

        except Exception as e:
            logger.error(f"Container execution failed: {e}")
            return ExecutionResult(success=False, error=str(e))
        finally:
            # Cleanup
            try:
                if "container" in locals():
                    container.stop()
            except Exception:
                pass

            # Cleanup working directory
            if work_path.exists():
                try:
                    import shutil

                    shutil.rmtree(work_path)
                except Exception:
                    logger.warning(f"Failed to cleanup working directory: {work_path}")

    def restart(self) -> None:
        """Restart the container (for persistent containers)."""
        # This would be useful for persistent containers
        # For now, we create fresh containers each time
        pass

    def stop(self) -> None:
        """Stop the container and cleanup resources."""
        # Cleanup is handled in the run method's finally block
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.stop()


# Register tool
registry.register("docker_sandbox", DockerSandboxRunner)
