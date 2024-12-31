# src/agents/decomposer.py

"""Task decomposition agent."""
import uuid
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json
from dataclasses import asdict
from ..core.data_structures import PrimitiveTask, ValidationResult
from ..core.exceptions import TaskDecompositionError
from .base import BaseAgent
from ..llm.manager import LLMManager
from ..utils.validation import validate_primitive_task
import z3
import os

class DecomposerAgent(BaseAgent):
    def __init__(self):
        """Initialize DecomposerAgent"""
        super().__init__(
            agent_id=str(uuid.uuid4()),
            agent_type="Decomposer"
        )
        self.llm_manager = LLMManager()
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.agent_id}")
        self.z3_solver = z3.Solver()
        self.known_primitives = set()

    async def decompose_task(self, task: Dict[str, Any]) -> List[PrimitiveTask]:
        """Decompose a high-level task into primitive tasks using LLM."""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="decompose",
                content={
                    "task": task
                }
            )
            decomposition = await self.llm_manager.parse_llm_response(llm_response)
            
            if not decomposition or "primitive_tasks" not in decomposition:
                self.logger.error("LLM response missing primitive_tasks")
                return []

            primitive_tasks = []
            for task_dict in decomposition["primitive_tasks"]:
                try:
                    primitive_task = PrimitiveTask(
                        task_id=task_dict.get("task_id", str(uuid.uuid4())),
                        type=task_dict.get("type", "shell_command"),
                        command=task_dict.get("command", ""),
                        working_directory=task_dict.get("working_directory", "./"),
                        environment=task_dict.get("environment", {}),
                        error_pattern=task_dict.get("error_pattern", ""),
                        success_pattern=task_dict.get("success_pattern", ""),
                        timeout_seconds=task_dict.get("timeout_seconds", 300),
                        retry_count=task_dict.get("retry_count", 0),
                        retry_delay_seconds=task_dict.get("retry_delay_seconds", 0),
                        depends_on=task_dict.get("depends_on", []),
                        cleanup_commands=task_dict.get("cleanup_commands", []),
                        validation_commands=task_dict.get("validation_commands", [])
                    )
                    primitive_tasks.append(primitive_task)
                except Exception as e:
                    self.logger.error(f"Failed to create primitive task: {str(e)}")
                    continue

            return primitive_tasks
        except Exception as e:
            self.logger.error(f"Task decomposition failed: {str(e)}")
            return []

    async def auto_fix_task(self, task: PrimitiveTask) -> Optional[PrimitiveTask]:
        """Attempt to automatically fix issues with a primitive task using LLM."""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="fix",
                content={
                    "task": asdict(task)
                }
            )
            fix_result = await self.llm_manager.parse_llm_response(llm_response)
            
            if not fix_result or "fixed_task" not in fix_result:
                self.logger.error("LLM response missing fixed_task")
                return None

            fixed_task_dict = fix_result["fixed_task"]
            return PrimitiveTask(
                task_id=fixed_task_dict.get("task_id", task.task_id),
                type=fixed_task_dict.get("type", task.type),
                command=fixed_task_dict.get("command", task.command),
                working_directory=fixed_task_dict.get("working_directory", task.working_directory),
                environment=fixed_task_dict.get("environment", task.environment),
                error_pattern=fixed_task_dict.get("error_pattern", task.error_pattern),
                success_pattern=fixed_task_dict.get("success_pattern", task.success_pattern),
                timeout_seconds=fixed_task_dict.get("timeout_seconds", task.timeout_seconds),
                retry_count=fixed_task_dict.get("retry_count", task.retry_count),
                retry_delay_seconds=fixed_task_dict.get("retry_delay_seconds", task.retry_delay_seconds),
                depends_on=fixed_task_dict.get("depends_on", task.depends_on),
                cleanup_commands=fixed_task_dict.get("cleanup_commands", task.cleanup_commands),
                validation_commands=fixed_task_dict.get("validation_commands", task.validation_commands)
            )
        except Exception as e:
            self.logger.error(f"Task auto-fix failed: {str(e)}")
            return None

    def is_primitive_description(self, task: dict) -> bool:
        """Check if a task description is already primitive"""
        required_fields = {"task_id", "type", "command", "working_directory"}
        return all(field in task for field in required_fields)

    def convert_to_primitive_task(self, task_dict: dict) -> PrimitiveTask:
        """Convert a dictionary to a PrimitiveTask object"""
        # Ensure all required fields are present with defaults
        task_dict = {
            "task_id": task_dict.get("task_id", task_dict.get("id", str(uuid.uuid4()))),
            "type": task_dict.get("type", "shell_command"),
            "command": task_dict.get("command", ""),
            "working_directory": task_dict.get("working_directory", "/tmp"),
            "environment": task_dict.get("environment", {"PATH": "/usr/bin"}),
            "error_pattern": task_dict.get("error_pattern", ".*error.*"),
            "success_pattern": task_dict.get("success_pattern", ".*success.*"),
            "timeout_seconds": task_dict.get("timeout_seconds", 30),
            "retry_count": task_dict.get("retry_count", 3),
            "retry_delay_seconds": task_dict.get("retry_delay_seconds", 5),
            "depends_on": task_dict.get("depends_on", []),
            "cleanup_commands": task_dict.get("cleanup_commands", []),
            "validation_commands": task_dict.get("validation_commands", [])
        }
        return PrimitiveTask(**task_dict)

    async def validate_decomposition(self,
                                   original_task: dict,
                                   primitive_tasks: List[PrimitiveTask]) -> List[str]:
        issues = []

        llm_response = await self.llm_manager.get_llm_response(
            prompt_type="validate",
            content={
                "original_task": original_task,
                "primitive_tasks": [asdict(task) for task in primitive_tasks]
            }
        )
        llm_validation = await self.llm_manager.parse_llm_response(llm_response)

        if not llm_validation.get("validation_results", {}).get("is_valid", False):
            issues.extend(llm_validation.get("validation_results", {}).get("issues", []))

        return issues

    def validate_dependencies(self, primitive_tasks: List[PrimitiveTask]) -> bool:
        s = z3.Solver()
        task_vars = {task.task_id: z3.Int(task.task_id) for task in primitive_tasks}
        
        for task in primitive_tasks:
            for dep_id in task.depends_on:
                if dep_id not in task_vars:
                    return False
                s.add(task_vars[task.task_id] > task_vars[dep_id])
        
        return s.check() == z3.sat

    def validate_environment(self, original_task: dict, primitive_tasks: List[PrimitiveTask]) -> bool:
        required_env = set(original_task.get('environment', {}).keys())
        provided_env = set()
        
        for task in primitive_tasks:
            provided_env.update(task.environment.keys())
        
        return required_env.issubset(provided_env)

    def validate_file_operations(self, original_task: dict, primitive_tasks: List[PrimitiveTask]) -> bool:
        if 'working_directory' in original_task:
            working_dir = original_task['working_directory']
            has_mkdir = any(
                task.type == 'file_operation' and 
                'mkdir' in task.command and 
                working_dir in task.command
                for task in primitive_tasks
            )
            return has_mkdir
        return True

    async def refine_decomposition(self, primitive_tasks: List[PrimitiveTask], validation_issues: List[str]) -> List[PrimitiveTask]:
        """Refine a task decomposition based on validation issues"""
        await self.log_action("refine_decomposition", f"Refining {len(primitive_tasks)} tasks with {len(validation_issues)} issues")

        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="fix_validation",
                content={
                    "tasks": [asdict(task) for task in primitive_tasks],
                    "issues": validation_issues
                }
            )
            refinement_result = await self.llm_manager.parse_llm_response(llm_response)

            refined_tasks = []
            for task_dict in refinement_result["refined_tasks"]:
                primitive_task = self.convert_to_primitive_task(task_dict)
                if validate_primitive_task(primitive_task):
                    refined_tasks.append(primitive_task)
                else:
                    fixed_task = await self.auto_fix_task(primitive_task)
                    if fixed_task:
                        refined_tasks.append(fixed_task)

            return refined_tasks

        except Exception as e:
            self.logger.error(f"Task refinement failed: {e}")
            return primitive_tasks  # Return original tasks if refinement fails