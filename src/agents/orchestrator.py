# src/agents/orchestrator.py

from .base import BaseAgent
from ..llm.manager import LLMManager
from ..core.data_structures import PrimitiveTask, State, ReActStep, ValidationResult
from typing import List, Dict, Optional, Any
import uuid
from dataclasses import asdict
import json
from datetime import datetime
import logging

from .validator import ValidatorAgent
from .decomposer import DecomposerAgent
from .error_detector import ErrorDetectorAgent
from .feedback import FeedbackAgent
from .state_manager import StateManager


class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            agent_type="Orchestrator"
        )
        self.state_manager = StateManager()
        self.decomposer = DecomposerAgent()
        self.llm_manager = LLMManager()
        self.validator = ValidatorAgent(llm_manager=self.llm_manager)
        self.error_detector = ErrorDetectorAgent()
        self.feedback = FeedbackAgent()
        self.react_history: List[ReActStep] = []

    async def orchestrate(self, high_level_tasks: List[dict]) -> Dict[str, Any]:
        """Orchestrate the entire process using ReAct pattern with LLM guidance"""
        try:
            # Initialize new state
            state = self.state_manager.create_new_state()
            
            # Get LLM's initial analysis of the high-level tasks
            initial_analysis = await self.get_llm_task_analysis(high_level_tasks)
            
            # Process tasks with LLM guidance
            primitive_tasks = []
            for task in high_level_tasks:
                react_step = await self.process_task_with_llm(
                    task, 
                    context=initial_analysis.get("task_contexts", {}).get(task.get("id"))
                )
                self.react_history.append(react_step)
                
                if react_step.reflection.startswith("SUCCESS"):
                    new_primitives = json.loads(react_step.reflection.split(":", 1)[1])
                    primitive_tasks.extend([
                        primitive for primitive in new_primitives 
                        if await self.validate_primitive(primitive)
                    ])

            # Validate entire DAG with LLM oversight
            validation_result = await self.validate_dag_with_llm(primitive_tasks)
            
            # Detect potential errors with LLM insights
            error_patterns = await self.error_detector.analyze_error_patterns(primitive_tasks)
            
            # Generate enhanced feedback
            feedback = await self.generate_enhanced_feedback(
                validation_result, 
                error_patterns,
                initial_analysis
            )
            
            # Prepare final output
            result = {
                'dag_id': str(uuid.uuid4()),
                'creation_time': datetime.utcnow().isoformat(),
                'primitive_tasks': [asdict(task) for task in primitive_tasks],
                'validation_result': asdict(validation_result),
                'error_analysis': error_patterns,
                'feedback': feedback,
                'llm_insights': initial_analysis.get("insights", []),
                'metadata': {
                    'total_tasks': len(primitive_tasks),
                    'validation_passed': validation_result.is_valid,
                    'error_count': sum(len(errors) for errors in error_patterns.values())
                }
            }
            
            # Update and save final state
            self.state_manager.update_state(state)
            return result

        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}")
            raise

    async def get_llm_task_analysis(self, tasks: List[dict]) -> Dict:
        """Get initial LLM analysis of the high-level tasks"""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="analyze_tasks",
                content={
                    "tasks": tasks,
                    "analysis_requirements": {
                        "identify_dependencies": True,
                        "suggest_decomposition_strategy": True,
                        "identify_critical_path": True,
                        "suggest_error_handling": True
                    }
                }
            )
            
            return await self.llm_manager.parse_llm_response(llm_response)
        
        except Exception as e:
            self.logger.error(f"LLM task analysis failed: {e}")
            return {"task_contexts": {}, "insights": []}

    async def process_task_with_llm(self, task: Dict, context: Optional[Dict] = None) -> ReActStep:
        """Process a task using LLM-based reasoning"""
        await self.log_action("process_task_with_llm", task)

        try:
            # Get LLM analysis of task
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="task_analysis",
                content=task
            )
            analysis_result = await self.llm_manager.parse_llm_response(llm_response)

            # Decompose task into primitives
            primitive_tasks = await self.decomposer.decompose_task(task)
            if not primitive_tasks:
                return ReActStep(
                    id=str(uuid.uuid4()),
                    thought="Task decomposition failed",
                    action="Task decomposition",
                    observation="No primitive tasks generated",
                    reflection="FAILURE: Task decomposition produced no tasks",
                    timestamp=datetime.now(),
                    agent_id=self.agent_id
                )

            # Validate primitive tasks
            valid_tasks = []
            for primitive in primitive_tasks:
                if await self.validate_primitive(asdict(primitive)):
                    valid_tasks.append(primitive)

            if not valid_tasks:
                return ReActStep(
                    id=str(uuid.uuid4()),
                    thought="Task validation failed",
                    action="Task validation",
                    observation="No valid primitive tasks",
                    reflection="FAILURE: No valid primitive tasks found",
                    timestamp=datetime.now(),
                    agent_id=self.agent_id
                )

            # Return success with valid primitive tasks
            return ReActStep(
                id=str(uuid.uuid4()),
                thought=analysis_result.get("thought", "Analyzing task requirements and dependencies"),
                action=analysis_result.get("action", "Decomposing task based on analysis"),
                observation=f"Generated {len(valid_tasks)} valid primitive tasks",
                reflection=f"SUCCESS:{json.dumps([asdict(t) for t in valid_tasks])}",
                timestamp=datetime.now(),
                agent_id=self.agent_id
            )

        except Exception as e:
            self.logger.error(f"Task processing failed: {e}")
            return ReActStep(
                id=str(uuid.uuid4()),
                thought="Error occurred during task processing",
                action="Error handling",
                observation=str(e),
                reflection="FAILURE: Task processing failed with error",
                timestamp=datetime.now(),
                agent_id=self.agent_id
            )

    async def get_llm_thought(self, task: dict, context: Optional[Dict]) -> str:
        """Get LLM's analysis and approach for processing a task"""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="task_analysis",
                content={"task": task, "context": context}
            )
            
            analysis = await self.llm_manager.parse_llm_response(llm_response)
            return analysis.get("thought", "No thought generated")
            
        except Exception as e:
            return f"Error getting LLM thought: {e}"

    async def get_llm_observation(self, 
                                original_task: dict,
                                primitive_tasks: List[PrimitiveTask],
                                validation_results: List[ValidationResult]) -> str:
        """Get LLM's observation of the decomposition results"""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="decomposition_analysis",
                content={
                    "original_task": original_task,
                    "primitive_tasks": [asdict(t) for t in primitive_tasks],
                    "validation_results": [asdict(v) for v in validation_results]
                }
            )
            
            analysis = await self.llm_manager.parse_llm_response(llm_response)
            return analysis.get("observation", "No observation generated")
            
        except Exception as e:
            return f"Error getting LLM observation: {e}"

    async def get_llm_reflection(self,
                               original_task: dict,
                               primitive_tasks: List[PrimitiveTask],
                               validation_results: List[ValidationResult]) -> str:
        """Get LLM's reflection and suggestions for improvement"""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="reflection",
                content={
                    "original_task": original_task,
                    "primitive_tasks": [asdict(t) for t in primitive_tasks],
                    "validation_results": [asdict(v) for v in validation_results]
                }
            )
            
            analysis = await self.llm_manager.parse_llm_response(llm_response)
            return analysis.get("reflection", "No reflection generated")
            
        except Exception as e:
            return f"Error getting LLM reflection: {e}"

    async def get_llm_insights(self, tasks: List[Dict]) -> List[str]:
        """Get insights from LLM about the tasks"""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="task_analysis",
                content={"tasks": tasks}
            )
            insights_result = await self.llm_manager.parse_llm_response(llm_response)
            return insights_result.get("insights", [])

        except Exception as e:
            self.logger.warning(f"Failed to get LLM insights: {e}")
            return []

    async def validate_primitive(self, primitive: dict) -> bool:
        """Validate a single primitive task with LLM oversight"""
        try:
            # Convert to PrimitiveTask object
            task = self.decomposer.convert_to_primitive_task(primitive)
            
            # Get LLM validation
            llm_validation = await self.validator.perform_llm_validation(task)
            
            return llm_validation.get("is_valid", False)
            
        except Exception as e:
            self.logger.error(f"Primitive validation failed: {e}")
            return False

    async def validate_dag_with_llm(self, tasks: List[PrimitiveTask]) -> ValidationResult:
        """Validate the DAG using both traditional and LLM-based validation"""
        await self.log_action("validate_dag_with_llm", f"Validating {len(tasks)} tasks")

        try:
            # 1. Traditional validation
            traditional_result = await self.validator.validate_dag(tasks)
            if not traditional_result.is_valid:
                return traditional_result

            # 2. LLM-based validation
            llm_validation = await self.perform_llm_validation(tasks)
            if not llm_validation["is_valid"]:
                return ValidationResult(
                    is_valid=False,
                    issues=llm_validation["issues"],
                    suggested_fixes=llm_validation["suggested_fixes"],
                    validation_time=datetime.now(),
                    validator_id=self.validator.validator_id
                )

            return ValidationResult(
                is_valid=True,
                issues=[],
                suggested_fixes=[],
                validation_time=datetime.now(),
                validator_id=self.validator.validator_id
            )

        except Exception as e:
            self.logger.error(f"DAG validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                issues=[str(e)],
                suggested_fixes=["Review DAG configuration"],
                validation_time=datetime.now(),
                validator_id=self.validator.validator_id
            )

    async def perform_llm_validation(self, tasks: List[PrimitiveTask]) -> Dict:
        """Perform LLM-based validation of the DAG"""
        try:
            response = await self.llm_manager.get_llm_response(
                prompt_type="validate_dag",
                content=[task.dict() for task in tasks]
            )
            return await self.llm_manager.parse_llm_response(response)

        except Exception as e:
            self.logger.error(f"LLM validation failed: {e}")
            return {
                "is_valid": False,
                "issues": [str(e)],
                "suggested_fixes": ["Review DAG configuration"]
            }

    async def log_action(self, action: str, details: any) -> None:
        """Log an action with details"""
        self.logger.info(f"{action}: {details}")

    async def generate_enhanced_feedback(
        self,
        validation_result: ValidationResult,
        error_patterns: Dict[str, Any],
        initial_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate enhanced feedback by combining validation results, error patterns, and initial analysis."""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="feedback",
                content={
                    "validation_result": asdict(validation_result),
                    "error_patterns": error_patterns,
                    "initial_analysis": initial_analysis
                }
            )
            feedback = await self.llm_manager.parse_llm_response(llm_response)
            return feedback
        except Exception as e:
            self.logger.error(f"Failed to generate enhanced feedback: {str(e)}")
            return {
                "feedback": [],
                "suggestions": [],
                "warnings": [str(e)]
            }

    async def generate_dag(self, input_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a DAG from input tasks."""
        try:
            result = await self.orchestrate(input_tasks)
            return result
        except Exception as e:
            self.logger.error(f"DAG generation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "dag": None
            }