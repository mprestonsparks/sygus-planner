# src/agents/error_detector.py

from .base import BaseAgent
from ..llm.manager import LLMManager
from ..core.data_structures import PrimitiveTask
from typing import List, Dict, Set, Any
import uuid
from dataclasses import asdict
import logging

class ErrorDetectorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            agent_type="ErrorDetector"
        )
        self.llm_manager = LLMManager()
        self.known_error_patterns: Set[str] = set()
        self.detected_errors: List[dict] = []
        self.logger = logging.getLogger(f"ErrorDetector_{str(uuid.uuid4())}")

    async def detect_potential_errors(self, task: PrimitiveTask) -> List[dict]:
        """Detect potential errors using both LLM and rule-based analysis"""
        await self.log_action("detect_errors", task.task_id)
        
        errors = []
        
        # 1. LLM-based error detection
        llm_errors = await self.perform_llm_error_detection(task)
        errors.extend(llm_errors)
        
        # 2. Rule-based error detection
        errors.extend(self.detect_resource_leaks(task))
        errors.extend(self.detect_race_conditions(task))
        errors.extend(self.detect_security_issues(task))
        
        # Store unique errors
        unique_errors = self.deduplicate_errors(errors)
        self.detected_errors.extend(unique_errors)
        
        return unique_errors

    async def perform_llm_error_detection(self, task: PrimitiveTask) -> List[dict]:
        """Use LLM to detect potential errors and edge cases"""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="error_detect",
                content=asdict(task),
                additional_context={
                    "error_categories": [
                        "resource_leaks",
                        "race_conditions",
                        "security_issues",
                        "network_failures",
                        "filesystem_errors",
                        "permission_issues",
                        "timing_issues"
                    ]
                }
            )
            
            error_analysis = await self.llm_manager.parse_llm_response(llm_response)
            return error_analysis["error_analysis"]
            
        except Exception as e:
            self.logger.error(f"LLM error detection failed: {e}")
            return [{
                "error_type": "llm_error",
                "description": f"LLM error detection failed: {str(e)}",
                "severity": "medium",
                "suggestion": "Fall back to rule-based error detection"
            }]

    def deduplicate_errors(self, errors: List[dict]) -> List[dict]:
        """Remove duplicate errors based on type and description"""
        seen_errors = set()
        unique_errors = []
        
        for error in errors:
            error_key = (error["error_type"], error["description"])
            if error_key not in seen_errors:
                seen_errors.add(error_key)
                unique_errors.append(error)
        
        return unique_errors

    async def analyze_patterns(self, tasks: List[Dict[str, Any]], errors: List[str]) -> Dict[str, Any]:
        """Analyze patterns in tasks and errors using LLM"""
        try:
            # Format input for LLM
            analysis_input = {
                "tasks": tasks,
                "errors": errors,
                "analysis_requirements": {
                    "identify_common_patterns": True,
                    "suggest_global_fixes": True,
                    "prioritize_issues": True
                }
            }
            
            # Get LLM response
            response = await self.llm_manager.get_llm_response("pattern_analysis", tasks=analysis_input)
            
            # Validate response has required fields
            if not response or not all(k in response for k in ["common_patterns", "anti_patterns", "optimization_opportunities"]):
                self.logger.error("LLM response missing pattern analysis")
                return {
                    "common_patterns": [],
                    "anti_patterns": [],
                    "optimization_opportunities": [],
                    "global_recommendations": []
                }
                
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to analyze patterns: {e}")
            return {
                "common_patterns": [],
                "anti_patterns": [],
                "optimization_opportunities": [],
                "global_recommendations": []
            }

    async def analyze_error_patterns(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns across tasks using LLM"""
        try:
            # Extract errors from tasks
            errors = []
            for task in tasks:
                if "errors" in task:
                    errors.extend(task["errors"])
                if "issues" in task:
                    errors.extend(task["issues"])

            # Analyze patterns if there are errors
            if errors:
                return await self.analyze_patterns(tasks, errors)
            else:
                return {
                    "common_patterns": [],
                    "anti_patterns": [],
                    "optimization_opportunities": [],
                    "global_recommendations": []
                }

        except Exception as e:
            self.logger.error(f"Failed to analyze error patterns: {e}")
            return {
                "common_patterns": [],
                "anti_patterns": [],
                "optimization_opportunities": [],
                "global_recommendations": []
            }

    async def generate_feedback(self, validation_result: Dict[str, Any], error_patterns: Dict[str, Any], initial_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced feedback using LLM"""
        try:
            # Format input for LLM
            feedback_input = {
                "validation_result": validation_result,
                "error_patterns": error_patterns,
                "initial_analysis": initial_analysis
            }
            
            # Get LLM response
            response = await self.llm_manager.get_llm_response("feedback", task=feedback_input)
            
            # Validate response has required fields
            if not response or not all(k in response for k in ["success", "issues", "recommendations", "priority_fixes"]):
                self.logger.error("LLM response missing feedback")
                return {
                    "success": False,
                    "issues": ["Invalid LLM response format"],
                    "recommendations": [],
                    "priority_fixes": []
                }
                
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to generate feedback: {e}")
            return {
                "success": False,
                "issues": [str(e)],
                "recommendations": [],
                "priority_fixes": []
            }

    async def log_action(self, action: str, details: any):
        """Log an action with details"""
        self.logger.info(f"Action: {action}, Details: {details}")