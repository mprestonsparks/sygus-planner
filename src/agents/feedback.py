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

    async def generate_feedback(self, task: Dict[str, Any], validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate feedback for a task using LLM"""
        try:
            # Format input for LLM
            feedback_input = {
                "task": task,
                "validation_results": validation_results,
                "feedback_requirements": {
                    "provide_suggestions": True,
                    "explain_issues": True,
                    "prioritize_fixes": True
                }
            }
            
            # Get LLM response
            response = await self.llm_manager.get_llm_response("feedback", task=feedback_input)
            
            # Validate response has required fields
            if not response or not all(k in response for k in ["feedback", "suggestions", "priority"]):
                self.logger.error("LLM response missing feedback")
                return {
                    "feedback": ["Invalid LLM response format"],
                    "suggestions": [],
                    "priority": "high",
                    "timestamp": datetime.now().isoformat()
                }
                
            # Add metadata
            response.update({
                "timestamp": datetime.now().isoformat()
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to generate feedback: {e}")
            return {
                "feedback": [str(e)],
                "suggestions": [],
                "priority": "high",
                "timestamp": datetime.now().isoformat()
            }

    async def analyze_feedback_history(self, task_id: str, feedback_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze feedback history for a task using LLM"""
        try:
            # Format input for LLM
            history_input = {
                "task_id": task_id,
                "feedback_history": feedback_history,
                "analysis_requirements": {
                    "identify_patterns": True,
                    "track_improvements": True,
                    "suggest_next_steps": True
                }
            }
            
            # Get LLM response
            response = await self.llm_manager.get_llm_response("feedback_history", task=history_input)
            
            # Validate response has required fields
            if not response or not all(k in response for k in ["patterns", "improvements", "next_steps"]):
                self.logger.error("LLM response missing history analysis")
                return {
                    "patterns": [],
                    "improvements": [],
                    "next_steps": [],
                    "timestamp": datetime.now().isoformat()
                }
                
            # Add metadata
            response.update({
                "timestamp": datetime.now().isoformat()
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to analyze feedback history: {e}")
            return {
                "patterns": [],
                "improvements": [],
                "next_steps": [],
                "timestamp": datetime.now().isoformat()
            }
