# src/agents/validator.py

from .base import BaseAgent
from ..llm.manager import LLMManager
from ..core.data_structures import PrimitiveTask, ValidationResult
from typing import List, Dict, Any
import uuid
from datetime import datetime
from dataclasses import asdict
import re
import logging
from typing import Tuple, Optional

class ValidatorAgent(BaseAgent):
    def __init__(self, llm_manager: LLMManager):
        """Initialize the validator agent"""
        super().__init__(
            agent_id=str(uuid.uuid4()),
            agent_type="Validator"
        )
        self.llm_manager = llm_manager
        self.validator_id = str(uuid.uuid4())
        self.validation_history: List[ValidationResult] = []
        self.logger = logging.getLogger(f"Validator_{id(self)}")

    async def log_action(self, action: str, details: any) -> None:
        """Log an action with details"""
        self.logger.info(f"{action}: {details}")

    def validate_task_structure(self, task: PrimitiveTask) -> Tuple[bool, List[str], List[str]]:
        """Validate the basic structure and required fields of a task"""
        issues = []
        fixes = []

        # Check required fields
        if not task.task_id:
            issues.append("Missing task ID")
            fixes.append("Add a unique task ID")

        if not task.type:
            issues.append("Missing task type")
            fixes.append("Specify task type (shell_command, api_call, file_operation)")

        if not task.command:
            issues.append("Missing command")
            fixes.append("Add command to execute")

        # Validate type-specific fields
        if task.type == "shell_command":
            shell_issues, shell_fixes = self.validate_shell_command(task)
            issues.extend(shell_issues)
            fixes.extend(shell_fixes)
        elif task.type == "api_call":
            api_issues, api_fixes = self.validate_api_call(task)
            issues.extend(api_issues)
            fixes.extend(api_fixes)
        elif task.type == "file_operation":
            file_issues, file_fixes = self.validate_file_operation(task)
            issues.extend(file_issues)
            fixes.extend(file_fixes)

        # Validate error handling
        error_issues, error_fixes = self.validate_error_handling(task)
        issues.extend(error_issues)
        fixes.extend(error_fixes)

        return len(issues) == 0, issues, fixes

    def validate_command(self, task: PrimitiveTask) -> Tuple[bool, List[str], List[str]]:
        """Validate command syntax and structure"""
        issues = []
        fixes = []

        if not task.command:
            issues.append("Empty command")
            fixes.append("Add command to execute")
            return False, issues, fixes

        # Basic command validation
        if task.type == "shell_command":
            if "|" in task.command or ">" in task.command or "<" in task.command:
                issues.append("Command contains shell redirections")
                fixes.append("Split complex commands into multiple tasks")

        elif task.type == "api_call":
            if not re.match(r'^(GET|POST|PUT|DELETE|PATCH)\s+https?://', task.command):
                issues.append("Invalid API call format")
                fixes.append("Use format: METHOD https://url")

        return len(issues) == 0, issues, fixes

    def validate_error_handling(self, task: PrimitiveTask) -> Tuple[List[str], List[str]]:
        """Validate error handling configuration of a task"""
        issues = []
        fixes = []

        # Validate error and success patterns
        if not task.error_pattern:
            issues.append("Missing error pattern")
            fixes.append("Add error pattern to detect failures")
        else:
            try:
                re.compile(task.error_pattern)
            except re.error:
                issues.append("Invalid error pattern regex")
                fixes.append("Fix error pattern regular expression")

        if not task.success_pattern:
            issues.append("Missing success pattern")
            fixes.append("Add success pattern to verify completion")
        else:
            try:
                re.compile(task.success_pattern)
            except re.error:
                issues.append("Invalid success pattern regex")
                fixes.append("Fix success pattern regular expression")

        # Validate timeout and retry settings
        if task.timeout_seconds <= 0:
            issues.append("Invalid timeout value")
            fixes.append("Set timeout_seconds to a positive value (recommended: 30-300)")

        if task.retry_count < 0:
            issues.append("Invalid retry count")
            fixes.append("Set retry_count to a non-negative value (recommended: 3)")

        if task.retry_delay_seconds < 0:
            issues.append("Invalid retry delay")
            fixes.append("Set retry_delay_seconds to a non-negative value (recommended: 5)")

        return issues, fixes

    def validate_shell_command(self, task: PrimitiveTask) -> Tuple[List[str], List[str]]:
        """Validate shell command specific requirements"""
        issues = []
        fixes = []

        if not task.working_directory:
            issues.append("Missing working directory")
            fixes.append("Set working directory")

        # Check for dangerous commands
        dangerous_patterns = [
            r"rm\s+-rf\s+/",
            r"dd\s+if=.*\s+of=/dev/",
            r"chmod\s+-R\s+777",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, task.command):
                issues.append(f"Potentially dangerous command pattern: {pattern}")
                fixes.append("Review and modify command to be safer")

        return issues, fixes

    def validate_api_call(self, task: PrimitiveTask) -> Tuple[List[str], List[str]]:
        """Validate API call specific requirements"""
        issues = []
        fixes = []

        # Check HTTP method
        if not re.match(r'^(GET|POST|PUT|DELETE|PATCH)\s+', task.command):
            issues.append("Missing or invalid HTTP method")
            fixes.append("Start command with valid HTTP method")

        # Check URL format
        if not re.search(r'https?://', task.command):
            issues.append("Invalid URL format")
            fixes.append("Use full URL with http:// or https://")

        # Check for auth headers if needed
        if "Authorization" not in task.environment and "auth" in task.command.lower():
            issues.append("Missing authorization header")
            fixes.append("Add Authorization to environment")

        return issues, fixes

    def validate_file_operation(self, task: PrimitiveTask) -> Tuple[List[str], List[str]]:
        """Validate file operation specific requirements"""
        issues = []
        fixes = []

        # Check working directory
        if not task.working_directory:
            issues.append("Missing working directory")
            fixes.append("Set working directory")

        # Check cleanup commands
        if not task.cleanup_commands:
            issues.append("Missing cleanup commands")
            fixes.append("Add cleanup commands to handle created files")

        # Check validation commands
        if not task.validation_commands:
            issues.append("Missing validation commands")
            fixes.append("Add commands to validate file operation success")

        return issues, fixes

    def validate_dag_structure(self, tasks: List[PrimitiveTask]) -> Tuple[bool, List[str], List[str]]:
        """Validate the DAG structure for cycles and dependencies"""
        issues = []
        fixes = []

        # Check for duplicate task IDs
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            issues.append("Duplicate task IDs found")
            fixes.append("Ensure all task IDs are unique")

        # Build dependency graph
        graph = {task.task_id: set(task.depends_on) for task in tasks}

        # Check for missing dependencies
        all_deps = set()
        for task in tasks:
            all_deps.update(task.depends_on)

        missing_deps = all_deps - set(task_ids)
        if missing_deps:
            issues.append(f"Missing dependencies: {missing_deps}")
            fixes.append("Add missing tasks or remove dependencies")

        # Check for cycles
        visited = set()
        temp_visited = set()

        def has_cycle(node: str) -> bool:
            if node in temp_visited:
                return True
            if node in visited:
                return False

            temp_visited.add(node)
            for dep in graph[node]:
                if has_cycle(dep):
                    return True
            temp_visited.remove(node)
            visited.add(node)
            return False

        for task_id in task_ids:
            if has_cycle(task_id):
                issues.append(f"Cyclic dependency detected involving task {task_id}")
                fixes.append("Remove cyclic dependencies")

        return len(issues) == 0, issues, fixes

    async def validate_task(self, task: PrimitiveTask) -> ValidationResult:
        """Validate a single primitive task using LLM."""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="validate_task",
                content={
                    "task": asdict(task)
                }
            )
            validation = await self.llm_manager.parse_llm_response(llm_response)
            
            if not validation or "validation_results" not in validation:
                self.logger.error("LLM response missing validation_results")
                return ValidationResult(
                    is_valid=False,
                    issues=["Invalid LLM response format"],
                    suggested_fixes=[],
                    validation_time=datetime.utcnow(),
                    validator_id=self.validator_id
                )

            results = validation["validation_results"]
            return ValidationResult(
                is_valid=results.get("is_valid", False),
                issues=results.get("issues", []),
                suggested_fixes=results.get("suggested_fixes", []),
                validation_time=datetime.utcnow(),
                validator_id=self.validator_id
            )
        except Exception as e:
            self.logger.error(f"Task validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                issues=[str(e)],
                suggested_fixes=[],
                validation_time=datetime.utcnow(),
                validator_id=self.validator_id
            )

    async def validate_dag(self, tasks: List[PrimitiveTask]) -> ValidationResult:
        """Validate a DAG of primitive tasks using LLM."""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="validate_dag",
                content={
                    "tasks": [asdict(task) for task in tasks]
                }
            )
            validation = await self.llm_manager.parse_llm_response(llm_response)
            
            if not validation or "validation_results" not in validation:
                self.logger.error("LLM response missing validation_results")
                return ValidationResult(
                    is_valid=False,
                    issues=["Invalid LLM response format"],
                    suggested_fixes=[],
                    validation_time=datetime.utcnow(),
                    validator_id=self.validator_id
                )

            results = validation["validation_results"]
            return ValidationResult(
                is_valid=results.get("is_valid", False),
                issues=results.get("issues", []),
                suggested_fixes=results.get("suggested_fixes", []),
                validation_time=datetime.utcnow(),
                validator_id=self.validator_id
            )
        except Exception as e:
            self.logger.error(f"DAG validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                issues=[str(e)],
                suggested_fixes=[],
                validation_time=datetime.utcnow(),
                validator_id=self.validator_id
            )
