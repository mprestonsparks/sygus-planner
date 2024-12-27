# src/agents/feedback.py

from .base import BaseAgent
from ..llm.manager import LLMManager
from ..core.data_structures import PrimitiveTask, ValidationResult
from typing import List, Dict, Any
import uuid
from datetime import datetime

class FeedbackAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            agent_type="Feedback"
        )
        self.llm_manager = LLMManager()

    async def generate_feedback(self, 
                              validation_result: ValidationResult,
                              error_patterns: Dict[str, List[dict]],
                              initial_analysis: Dict) -> Dict[str, Any]:
        """Generate feedback based on validation results and error patterns"""
        try:
            llm_response = await self.llm_manager.get_llm_response(
                prompt_type="generate_feedback",
                content={
                    "validation_result": validation_result,
                    "error_patterns": error_patterns,
                    "initial_analysis": initial_analysis
                }
            )
            
            feedback = await self.llm_manager.parse_llm_response(llm_response)
            return feedback
            
        except Exception as e:
            self.logger.error(f"Feedback generation failed: {e}")
            return {
                "summary": "Feedback generation failed",
                "suggestions": [],
                "error": str(e)
            }
