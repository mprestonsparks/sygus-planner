"""LLM Manager for handling interactions with language models."""
import asyncio
import json
import logging
import re
from typing import Dict, Any, Optional
from dataclasses import asdict
from anthropic import AsyncAnthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from ..core.exceptions import LLMError
from ..core.data_structures import LLMResponse
from ..core.config import get_settings


class LLMManager:
    def __init__(self):
        """Initialize LLM Manager"""
        settings = get_settings()
        self.model = ChatAnthropic(
            model=settings.model_name,
            anthropic_api_key=settings.anthropic_api_key
        )
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        
        self.prompts = {
            "decompose": {
                "system": "You are a task decomposition expert. Your role is to break down high-level tasks into atomic, executable primitive tasks.",
                "user": "Please decompose the following task into primitive tasks:\n{task}"
            },
            "validate": {
                "system": "You are a task validation expert. Your role is to validate primitive tasks for correctness and completeness.",
                "user": "Please validate the following task:\n{task}"
            },
            "validate_task": {
                "system": "You are a task validation expert. Your role is to validate primitive tasks for correctness and completeness.",
                "user": "Please validate the following task:\n{task}"
            },
            "validate_dag": {
                "system": "You are a DAG validation expert. Your role is to validate task dependencies and execution order.",
                "user": "Please validate the following task DAG:\n{tasks}"
            },
            "fix": {
                "system": "You are a task fixing expert. Your role is to fix issues in primitive tasks.",
                "user": "Please fix the following task:\n{task}"
            },
            "analyze_tasks": {
                "system": "You are a task analysis expert. Your role is to analyze tasks for potential issues and optimizations.",
                "user": "Please analyze the following tasks:\n{tasks}"
            },
            "errors": {
                "system": "You are an error pattern detection expert. Your role is to identify potential error patterns in tasks.",
                "user": "Please analyze the following tasks for error patterns:\n{tasks}"
            },
            "feedback": {
                "system": "You are a feedback generation expert. Your role is to provide enhanced feedback on task execution.",
                "user": "Please generate feedback for the following:\n{task}"
            },
            "pattern_analysis": {
                "system": """You are an error pattern analysis expert. Your role is to analyze error patterns and provide insights.
                
                Analysis Categories:
                1. Pattern Types:
                   - Identify common error patterns
                   - Group similar errors
                   - Calculate pattern frequencies
                
                2. Root Causes:
                   - Identify underlying causes
                   - Find common triggers
                   - Suggest preventive measures
                
                3. Impact Analysis:
                   - Calculate error severity
                   - Assess business impact
                   - Prioritize fixes
                
                4. Recommendations:
                   - Suggest mitigation strategies
                   - Propose monitoring solutions
                   - Recommend process improvements
                
                Output Format:
                {
                    "patterns": [
                        {
                            "type": str,
                            "description": str,
                            "frequency": int,
                            "severity": str,
                            "root_cause": str,
                            "mitigation": str
                        }
                    ],
                    "summary": {
                        "total_errors": int,
                        "critical_patterns": List[str],
                        "recommended_actions": List[str]
                    }
                }""",
                "user": "Please analyze the following error patterns: {errors}"
            }
        }

    def get_prompt(self, prompt_type: str, content: Dict) -> str:
        """Get prompt for LLM"""
        try:
            prompt_template = self.prompts[prompt_type]
            prompt = f"{prompt_template['system']}\n\n{prompt_template['user']}"
            
            # Format the prompt with the content
            try:
                return prompt.format(**content)
            except KeyError as e:
                self.logger.warning(f"Missing key in content: {e}")
                return prompt.format(**{k: content.get(k, '') for k in re.findall(r'\{(\w+)\}', prompt)})
            
        except Exception as e:
            self.logger.error(f"Failed to get prompt: {e}")
            raise LLMError(f"Failed to get prompt: {e}")

    async def get_llm_response(self, prompt_type: str, content: Any) -> str:
        """Get a response from the LLM"""
        try:
            prompt = self.prompts[prompt_type]
            system_prompt = prompt["system"]
            
            # Format content for user prompt
            if isinstance(content, (dict, list)):
                formatted_content = json.dumps(content, indent=2)
            else:
                formatted_content = str(content)
                
            user_prompt = prompt["user"].format(task=formatted_content, tasks=formatted_content)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.model.agenerate([messages])
            return response.generations[0][0].text

        except Exception as e:
            self.logger.error(f"LLM request failed: {e}")
            raise LLMError(f"Failed to get LLM response: {e}")

    async def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        try:
            # Try to parse as JSON first
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass

            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*({[^`]*})\s*```', response)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Fall back to basic parsing with default values
            return {
                "primitive_tasks": [],
                "validation_results": {
                    "is_valid": False,
                    "issues": [],
                    "fixes": []
                },
                "fixed_task": None,
                "insights": [],
                "errors": []
            }

        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return {
                "primitive_tasks": [],
                "validation_results": {
                    "is_valid": False,
                    "issues": [f"Failed to parse response: {str(e)}"],
                    "fixes": []
                },
                "fixed_task": None,
                "insights": [],
                "errors": [str(e)]
            }

    async def log_action(self, action: str, details: any):
        """Log an action with details"""
        self.logger.info(f"Action: {action}, Details: {details}")