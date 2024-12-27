# src/agents/decomposer.py

from typing import List, Dict, Optional
import z3
from datetime import datetime
import uuid
import json
from dataclasses import asdict

from ..core.data_structures import PrimitiveTask, ValidationResult
from ..core.exceptions import TaskDecompositionError
from .base import BaseAgent
from ..llm.manager import LLMManager
from ..utils.validation import validate_primitive_task

class DecomposerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            agent_type="Decomposer"
        )
        self.llm_manager = LLMManager()
        self.z3_solver = z3.Solver()
        self.known_primitives = set()

    async def decompose_task(self, task: dict) -> List[PrimitiveTask]:
        await self.log_action("decompose_task", task)
        
        try:
            if self.is_primitive_description(task):
                primitive_task = self.convert_to_primitive_task(task)
                if validate_primitive_task(primitive_task):
                    return [primitive_task]
            
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="decompose",
                content=task
            )
            decomposition_result = await self.llm_manager.parse_llm_response(llm_response)
            
            primitive_tasks = []
            for task_dict in decomposition_result["primitive_tasks"]:
                primitive_task = self.convert_to_primitive_task(task_dict)
                if validate_primitive_task(primitive_task):
                    primitive_tasks.append(primitive_task)
                else:
                    fixed_task = await self.auto_fix_task(primitive_task)
                    if fixed_task:
                        primitive_tasks.append(fixed_task)
            
            validation_issues = await self.validate_decomposition(task, primitive_tasks)
            
            if validation_issues:
                refined_tasks = await self.refine_decomposition(primitive_tasks, validation_issues)
                return refined_tasks
            
            return primitive_tasks

        except Exception as e:
            self.logger.error(f"Task decomposition failed: {e}")
            raise TaskDecompositionError(f"Failed to decompose task: {e}")

    def is_primitive_description(self, task_description: dict) -> bool:
        required_fields = {
            'type', 'command', 'working_directory', 'environment',
            'error_pattern', 'success_pattern', 'timeout_seconds'
        }
        return all(field in task_description for field in required_fields)

    def convert_to_primitive_task(self, description: dict) -> PrimitiveTask:
        return PrimitiveTask(
            task_id=description.get('task_id', f"TASK_{uuid.uuid4().hex[:8]}"),
            type=description['type'],
            command=description['command'],
            working_directory=description['working_directory'],
            environment=description['environment'],
            error_pattern=description['error_pattern'],
            success_pattern=description['success_pattern'],
            timeout_seconds=description['timeout_seconds'],
            retry_count=description.get('retry_count', 3),
            retry_delay_seconds=description.get('retry_delay_seconds', 5),
            depends_on=description.get('depends_on', []),
            cleanup_commands=description.get('cleanup_commands', []),
            validation_commands=description.get('validation_commands', [])
        )

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
        
        for validation_result in llm_validation["validation_results"]:
            if not validation_result["is_valid"]:
                issues.extend(validation_result["issues"])
        
        if not self.validate_dependencies(primitive_tasks):
            issues.append("Invalid dependency structure detected")
        
        if not self.validate_environment(original_task, primitive_tasks):
            issues.append("Missing environment variables")
        
        if not self.validate_file_operations(original_task, primitive_tasks):
            issues.append("Invalid file operations detected")
        
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

    async def auto_fix_task(self, task: PrimitiveTask) -> Optional[PrimitiveTask]:
        fixed_task = PrimitiveTask(**asdict(task))
        
        if not Path(fixed_task.working_directory).is_absolute():
            fixed_task.working_directory = str(Path(fixed_task.working_directory).absolute())
        
        if fixed_task.type == 'shell_command' and 'PATH' not in fixed_task.environment:
            fixed_task.environment['PATH'] = '/usr/local/bin:/usr/bin:/bin'
        
        if fixed_task.timeout_seconds <= 0:
            fixed_task.timeout_seconds = 30 if fixed_task.type == 'api_call' else 60
        
        if not fixed_task.cleanup_commands and fixed_task.type == 'file_operation':
            fixed_task.cleanup_commands = [
                f'rm -f {fixed_task.working_directory}/*.tmp',
                f'rm -f {fixed_task.working_directory}/*.lock'
            ]
        
        return fixed_task if validate_primitive_task(fixed_task) else None