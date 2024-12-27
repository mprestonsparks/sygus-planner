# src/agents/error_detector.py

from .base import BaseAgent
from ..llm.manager import LLMManager
from ..core.data_structures import PrimitiveTask
from typing import List, Dict, Set
import uuid
from dataclasses import asdict

class ErrorDetectorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            agent_type="ErrorDetector"
        )
        self.llm_manager = LLMManager()
        self.known_error_patterns: Set[str] = set()
        self.detected_errors: List[dict] = []

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

    async def analyze_error_patterns(self, tasks: List[PrimitiveTask]) -> Dict[str, List[dict]]:
        """Analyze error patterns across all tasks using both LLM and rule-based analysis"""
        await self.log_action("analyze_patterns", f"Analyzing {len(tasks)} tasks")
        
        # 1. Collect errors from all tasks
        all_errors = []
        for task in tasks:
            task_errors = await self.detect_potential_errors(task)
            all_errors.extend(task_errors)
        
        # 2. Use LLM to analyze patterns in collected errors
        pattern_analysis = await self.perform_llm_pattern_analysis(all_errors)
        
        # 3. Combine with traditional pattern analysis
        error_patterns = self.group_errors_by_type(all_errors)
        
        # 4. Enhance patterns with LLM insights
        enhanced_patterns = self.combine_pattern_analyses(error_patterns, pattern_analysis)
        
        return enhanced_patterns

    async def perform_llm_pattern_analysis(self, errors: List[dict]) -> Dict:
        """Use LLM to analyze patterns in collected errors"""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="pattern_analysis",
                content={
                    "errors": errors,
                    "analysis_requirements": {
                        "identify_common_patterns": True,
                        "suggest_global_fixes": True,
                        "prioritize_issues": True
                    }
                }
            )
            
            return await self.llm_manager.parse_llm_response(llm_response)
            
        except Exception as e:
            self.logger.error(f"LLM pattern analysis failed: {e}")
            return {
                "patterns": [],
                "global_fixes": [],
                "priorities": []
            }

    def combine_pattern_analyses(self, 
                               rule_based_patterns: Dict[str, List[dict]], 
                               llm_patterns: Dict) -> Dict[str, List[dict]]:
        """Combine rule-based and LLM-based pattern analyses"""
        combined_patterns = rule_based_patterns.copy()
        
        # Enhance with LLM insights
        for pattern in llm_patterns.get("patterns", []):
            pattern_type = pattern.get("type")
            if pattern_type in combined_patterns:
                # Add LLM insights to existing pattern type
                combined_patterns[pattern_type].extend(pattern.get("insights", []))
            else:
                # Create new pattern type from LLM analysis
                combined_patterns[pattern_type] = pattern.get("insights", [])
        
        return combined_patterns