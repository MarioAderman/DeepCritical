"""
Docker sandbox data types for DeepCritical research workflows.

This module defines Pydantic models for Docker sandbox operations including
configuration, execution requests, results, and execution policies.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class DockerSandboxPolicies(BaseModel):
    """Execution policies for different languages in Docker sandbox."""

    bash: bool = Field(True, description="Allow bash execution")
    shell: bool = Field(True, description="Allow shell execution")
    sh: bool = Field(True, description="Allow sh execution")
    pwsh: bool = Field(True, description="Allow PowerShell execution")
    powershell: bool = Field(True, description="Allow PowerShell execution")
    ps1: bool = Field(True, description="Allow ps1 execution")
    python: bool = Field(True, description="Allow Python execution")
    javascript: bool = Field(False, description="Allow JavaScript execution")
    html: bool = Field(False, description="Allow HTML execution")
    css: bool = Field(False, description="Allow CSS execution")

    def is_language_allowed(self, language: str) -> bool:
        """Check if a language is allowed for execution."""
        language_lower = language.lower()
        return getattr(self, language_lower, False)

    def get_allowed_languages(self) -> List[str]:
        """Get list of allowed languages."""
        allowed = []
        for field_name in self.__fields__:
            if getattr(self, field_name):
                allowed.append(field_name)
        return allowed

    class Config:
        json_schema_extra = {
            "example": {
                "bash": True,
                "shell": True,
                "python": True,
                "javascript": False,
                "html": False,
            }
        }


class DockerSandboxEnvironment(BaseModel):
    """Environment variables and settings for Docker sandbox."""

    variables: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables"
    )
    working_directory: str = Field(
        "/workspace", description="Working directory in container"
    )
    user: Optional[str] = Field(None, description="User to run as")
    network_mode: Optional[str] = Field(None, description="Network mode for container")

    def add_variable(self, key: str, value: str) -> None:
        """Add an environment variable."""
        self.variables[key] = value

    def remove_variable(self, key: str) -> bool:
        """Remove an environment variable."""
        if key in self.variables:
            del self.variables[key]
            return True
        return False

    def get_variable(self, key: str, default: str = "") -> str:
        """Get an environment variable value."""
        return self.variables.get(key, default)

    class Config:
        json_schema_extra = {
            "example": {
                "variables": {"PYTHONUNBUFFERED": "1", "PATH": "/usr/local/bin"},
                "working_directory": "/workspace",
                "user": "sandbox",
            }
        }


class DockerSandboxConfig(BaseModel):
    """Configuration for Docker sandbox settings."""

    image: str = Field("python:3.11-slim", description="Docker image to use")
    working_directory: str = Field(
        "/workspace", description="Working directory in container"
    )
    cpu_limit: Optional[float] = Field(None, description="CPU limit (cores)")
    memory_limit: Optional[str] = Field(
        None, description="Memory limit (e.g., '512m', '1g')"
    )
    auto_remove: bool = Field(
        True, description="Automatically remove container after execution"
    )
    network_disabled: bool = Field(False, description="Disable network access")
    privileged: bool = Field(False, description="Run container in privileged mode")
    volumes: Dict[str, str] = Field(
        default_factory=dict, description="Volume mounts (host_path:container_path)"
    )

    def add_volume(self, host_path: str, container_path: str) -> None:
        """Add a volume mount."""
        self.volumes[host_path] = container_path

    def remove_volume(self, host_path: str) -> bool:
        """Remove a volume mount."""
        if host_path in self.volumes:
            del self.volumes[host_path]
            return True
        return False

    class Config:
        json_schema_extra = {
            "example": {
                "image": "python:3.11-slim",
                "working_directory": "/workspace",
                "cpu_limit": 1.0,
                "memory_limit": "512m",
                "auto_remove": True,
                "volumes": {"/host/data": "/workspace/data"},
            }
        }


class DockerExecutionRequest(BaseModel):
    """Request parameters for Docker execution."""

    language: str = Field(
        "python", description="Programming language (python, bash, shell, etc.)"
    )
    code: str = Field("", description="Code string to execute")
    command: Optional[str] = Field(
        None, description="Explicit command to run (overrides code)"
    )
    environment: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables"
    )
    timeout: int = Field(60, description="Execution timeout in seconds")
    execution_policy: Optional[Dict[str, bool]] = Field(
        None, description="Custom execution policies for languages"
    )
    files: Dict[str, str] = Field(
        default_factory=dict, description="Files to create in container"
    )

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v):
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        """Validate language is not empty."""
        if not v or not v.strip():
            raise ValueError("Language cannot be empty")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "language": "python",
                "code": "print('Hello, World!')",
                "timeout": 30,
                "environment": {"PYTHONUNBUFFERED": "1"},
                "execution_policy": {"python": True, "bash": True},
            }
        }


class DockerExecutionResult(BaseModel):
    """Result from Docker execution."""

    success: bool = Field(..., description="Whether execution was successful")
    stdout: str = Field("", description="Standard output")
    stderr: str = Field("", description="Standard error")
    exit_code: int = Field(..., description="Exit code")
    files_created: List[str] = Field(
        default_factory=list, description="Files created during execution"
    )
    execution_time: float = Field(0.0, description="Execution time in seconds")
    error_message: Optional[str] = Field(
        None, description="Error message if execution failed"
    )

    @property
    def output(self) -> str:
        """Get combined output (stdout + stderr)."""
        return f"{self.stdout}\n{self.stderr}".strip()

    def is_timeout(self) -> bool:
        """Check if execution timed out."""
        return self.exit_code == 124

    def has_error(self) -> bool:
        """Check if execution had an error."""
        return not self.success or self.exit_code != 0

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "stdout": "Hello, World!",
                "stderr": "",
                "exit_code": 0,
                "files_created": ["/workspace/script.py"],
                "execution_time": 0.5,
            }
        }


class DockerSandboxContainerInfo(BaseModel):
    """Information about the Docker container used for execution."""

    container_id: str = Field(..., description="Container ID")
    container_name: str = Field(..., description="Container name")
    image: str = Field(..., description="Docker image used")
    status: str = Field(..., description="Container status")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    started_at: Optional[str] = Field(None, description="Start timestamp")
    finished_at: Optional[str] = Field(None, description="Finish timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "container_id": "abc123...",
                "container_name": "deepcritical-sandbox-abc123",
                "image": "python:3.11-slim",
                "status": "exited",
            }
        }


class DockerSandboxMetrics(BaseModel):
    """Metrics for Docker sandbox operations."""

    total_executions: int = Field(0, description="Total executions")
    successful_executions: int = Field(0, description="Successful executions")
    failed_executions: int = Field(0, description="Failed executions")
    average_execution_time: float = Field(0.0, description="Average execution time")
    total_cpu_time: float = Field(0.0, description="Total CPU time used")
    total_memory_used: float = Field(0.0, description="Total memory used")
    containers_created: int = Field(0, description="Containers created")
    containers_reused: int = Field(0, description="Containers reused")

    def record_execution(self, result: DockerExecutionResult) -> None:
        """Record an execution result."""
        self.total_executions += 1
        if result.success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        # Update average execution time
        if self.total_executions == 1:
            self.average_execution_time = result.execution_time
        else:
            self.average_execution_time = (
                (self.average_execution_time * (self.total_executions - 1))
                + result.execution_time
            ) / self.total_executions

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    class Config:
        json_schema_extra = {
            "example": {
                "total_executions": 100,
                "successful_executions": 95,
                "failed_executions": 5,
                "average_execution_time": 1.2,
                "success_rate": 0.95,
            }
        }


class DockerSandboxRequest(BaseModel):
    """Complete request for Docker sandbox operations."""

    execution: DockerExecutionRequest = Field(..., description="Execution parameters")
    config: Optional[DockerSandboxConfig] = Field(
        None, description="Sandbox configuration"
    )
    environment: Optional[DockerSandboxEnvironment] = Field(
        None, description="Environment settings"
    )
    policies: Optional[DockerSandboxPolicies] = Field(
        None, description="Execution policies"
    )

    def get_config(self) -> DockerSandboxConfig:
        """Get the Docker sandbox configuration."""
        return self.config or DockerSandboxConfig()

    def get_environment(self) -> DockerSandboxEnvironment:
        """Get the Docker sandbox environment."""
        return self.environment or DockerSandboxEnvironment()

    def get_policies(self) -> DockerSandboxPolicies:
        """Get the Docker sandbox policies."""
        return self.policies or DockerSandboxPolicies()

    class Config:
        json_schema_extra = {
            "example": {
                "execution": {
                    "language": "python",
                    "code": "print('Hello, World!')",
                    "timeout": 30,
                },
                "config": {
                    "image": "python:3.11-slim",
                    "auto_remove": True,
                },
                "environment": {
                    "variables": {"PYTHONUNBUFFERED": "1"},
                    "working_directory": "/workspace",
                },
            }
        }


class DockerSandboxResponse(BaseModel):
    """Complete response from Docker sandbox operations."""

    request: DockerSandboxRequest = Field(..., description="Original request")
    result: DockerExecutionResult = Field(..., description="Execution result")
    container_info: Optional[DockerSandboxContainerInfo] = Field(
        None, description="Container information"
    )
    metrics: Optional[DockerSandboxMetrics] = Field(
        None, description="Execution metrics"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "request": {},
                "result": {
                    "success": True,
                    "stdout": "Hello, World!",
                    "exit_code": 0,
                    "execution_time": 0.5,
                },
                "container_info": {
                    "container_id": "abc123...",
                    "container_name": "deepcritical-sandbox-abc123",
                    "image": "python:3.11-slim",
                },
                "metrics": {
                    "total_executions": 1,
                    "successful_executions": 1,
                    "average_execution_time": 0.5,
                },
            }
        }


# Handle forward references for Pydantic v2
DockerSandboxConfig.model_rebuild()
DockerExecutionRequest.model_rebuild()
DockerExecutionResult.model_rebuild()
DockerSandboxEnvironment.model_rebuild()
DockerSandboxPolicies.model_rebuild()
DockerSandboxContainerInfo.model_rebuild()
DockerSandboxMetrics.model_rebuild()
DockerSandboxRequest.model_rebuild()
DockerSandboxResponse.model_rebuild()
