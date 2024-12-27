# src/agents/validator.py

from .base import BaseAgent
from ..llm.manager import LLMManager
from ..core.data_structures import PrimitiveTask, ValidationResult
from typing import List, Dict, Any
import uuid
from datetime import datetime
from dataclasses import asdict

class ValidatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            agent_type="Validator"
        )
        self.llm_manager = LLMManager()
        self.validation_history: List[ValidationResult] = []

    async def validate_task(self, task: PrimitiveTask) -> ValidationResult:
        """Validate a single primitive task using both LLM and rule-based validation"""
        await self.log_action("validate_task", task.task_id)
        
        # Collect issues and fixes from both validation approaches
        issues = []
        fixes = []

        # 1. LLM-based semantic validation
        llm_validation = await self.perform_llm_validation(task)
        if not llm_validation["is_valid"]:
            issues.extend(llm_validation["issues"])
            fixes.extend(llm_validation["suggested_fixes"])

        # 2. Traditional rule-based validation
        structure_validation = self.validate_task_structure(task)
        if not structure_validation[0]:
            issues.extend(structure_validation[1])
            fixes.extend(structure_validation[2])

        command_validation = self.validate_command(task)
        if not command_validation[0]:
            issues.extend(command_validation[1])
            fixes.extend(command_validation[2])

        error_validation = self.validate_error_handling(task)
        if not error_validation[0]:
            issues.extend(error_validation[1])
            fixes.extend(error_validation[2])

        result = ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            suggested_fixes=fixes,
            validation_time=datetime.utcnow(),
            validator_id=self.agent_id
        )
        
        self.validation_history.append(result)
        return result

    async def perform_llm_validation(self, task: PrimitiveTask) -> Dict:
        """Use LLM to perform semantic validation of a task"""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="validate",
                content=asdict(task),
                additional_context={
                    "validation_focus": [
                        "atomicity",
                        "completeness",
                        "correctness",
                        "safety",
                        "idempotency"
                    ]
                }
            )
            
            validation_result = await self.llm_manager.parse_llm_response(llm_response)
            return validation_result["validation_results"][0]  # Take first result since we're validating a single task
            
        except Exception as e:
            self.logger.error(f"LLM validation failed: {e}")
            return {
                "is_valid": False,
                "issues": [f"LLM validation failed: {str(e)}"],
                "suggested_fixes": ["Retry validation or fall back to rule-based validation"]
            }

    async def validate_dag(self, tasks: List[PrimitiveTask]) -> ValidationResult:
        """Validate the entire DAG using both LLM and rule-based validation"""
        await self.log_action("validate_dag", f"Validating {len(tasks)} tasks")
        
        issues = []
        fixes = []

        # 1. LLM-based DAG validation
        llm_dag_validation = await self.perform_llm_dag_validation(tasks)
        if not llm_dag_validation["is_valid"]:
            issues.extend(llm_dag_validation["issues"])
            fixes.extend(llm_dag_validation["suggested_fixes"])

        # 2. Traditional DAG validation
        dag_structure_validation = self.validate_dag_structure(tasks)
        if not dag_structure_validation[0]:
            issues.extend(dag_structure_validation[1])
            fixes.extend(dag_structure_validation[2])

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            suggested_fixes=fixes,
            validation_time=datetime.utcnow(),
            validator_id=self.agent_id
        )

    async def perform_llm_dag_validation(self, tasks: List[PrimitiveTask]) -> Dict:
        """Use LLM to validate the entire DAG structure and relationships"""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="validate_dag",
                content={
                    "tasks": [asdict(task) for task in tasks],
                    "validation_requirements": {
                        "check_completeness": True,
                        "check_dependencies": True,
                        "check_resource_conflicts": True,
                        "check_execution_order": True
                    }
                }
            )
            
            return await self.llm_manager.parse_llm_response(llm_response)
            
        except Exception as e:
            self.logger.error(f"LLM DAG validation failed: {e}")
            return {
                "is_valid": False,
                "issues": [f"LLM DAG validation failed: {str(e)}"],
                "suggested_fixes": ["Review DAG structure manually"]
            }
