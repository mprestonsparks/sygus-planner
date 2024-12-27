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
        self.validator = ValidatorAgent()
        self.error_detector = ErrorDetectorAgent()
        self.feedback = FeedbackAgent()
        self.llm_manager = LLMManager()
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

    async def process_task_with_llm(self, 
                                  task: dict, 
                                  context: Optional[Dict] = None) -> ReActStep:
        """Process a single task using ReAct pattern with LLM guidance"""
        react_step = ReActStep.create(self.agent_id)
        
        try:
            # THOUGHT: Get LLM's analysis of the task
            react_step.thought = await self.get_llm_thought(task, context)
            
            # ACTION: Decompose based on LLM guidance
            react_step.action = "Decomposing task based on LLM analysis"
            primitive_tasks = await self.decomposer.decompose_task(task)
            
            # OBSERVATION: Validate decomposition
            validation_results = []
            for p_task in primitive_tasks:
                validation_result = await self.validator.validate_task(p_task)
                validation_results.append(validation_result)
            
            react_step.observation = await self.get_llm_observation(
                task, primitive_tasks, validation_results
            )
            
            # REFLECTION: Get LLM's reflection and suggestions
            react_step.reflection = await self.get_llm_reflection(
                task, primitive_tasks, validation_results
            )
            
            if all(v.is_valid for v in validation_results):
                react_step.reflection = f"SUCCESS:{json.dumps([asdict(t) for t in primitive_tasks])}"
            
            return react_step

        except Exception as e:
            react_step.reflection = f"ERROR:{str(e)}"
            return react_step

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
        """Validate the entire DAG with LLM oversight"""
        # Get both traditional and LLM validation
        traditional_result = await self.validator.validate_dag(tasks)
        llm_result = await self.validator.perform_llm_dag_validation(tasks)
        
        # Combine results
        all_issues = traditional_result.issues + llm_result.get("issues", [])
        all_fixes = traditional_result.suggested_fixes + llm_result.get("suggested_fixes", [])
        
        return ValidationResult(
            is_valid=traditional_result.is_valid and llm_result.get("is_valid", False),
            issues=list(set(all_issues)),
            suggested_fixes=list(set(all_fixes)),
            validation_time=datetime.utcnow(),
            validator_id=self.agent_id
        )

    async def generate_enhanced_feedback(self,
                                      validation_result: ValidationResult,
                                      error_patterns: Dict[str, List[dict]],
                                      initial_analysis: Dict) -> Dict:
        """Generate enhanced feedback with LLM insights"""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="generate_feedback",
                content={
                    "validation_result": asdict(validation_result),
                    "error_patterns": error_patterns,
                    "initial_analysis": initial_analysis
                }
            )
            
            feedback = await self.llm_manager.parse_llm_response(llm_response)
            return feedback
            
        except Exception as e:
            self.logger.error(f"Enhanced feedback generation failed: {e}")
            return {
                "summary": "Feedback generation failed",
                "issues": [],
                "recommendations": []
            }