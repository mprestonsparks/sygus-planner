"""LLM Manager for handling interactions with language models."""
import asyncio
import json
import logging
import re
from typing import Dict, Any, Optional, Union
from dataclasses import asdict
from anthropic import AsyncAnthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from ..core.exceptions import LLMError
from ..core.data_structures import LLMResponse
from ..core.config import get_settings
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class LLMManager:
    def __init__(self):
        """Initialize LLM Manager"""
        settings = get_settings()
        self.model = ChatAnthropic(
            model=settings.model_name,
            anthropic_api_key=settings.anthropic_api_key,
            max_retries=2,
            model_kwargs={"max_tokens": 4000}  # Set max tokens instead of timeout
        )
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        self._client = None
        self._semaphore = asyncio.Semaphore(3)  # Limit concurrent requests
        
        self.prompts = {
            "decompose": {
                "system": """You are a task decomposition expert. Given a complex task, break it down into primitive tasks that can be directly executed. Each primitive task should be atomic and independently executable.

Your response must be a valid JSON object with the following structure:
{
    "subtasks": [
        {
            "task_id": string,
            "type": string,
            "command": string,
            "working_directory": string,
            "environment": object,
            "error_pattern": string,
            "success_pattern": string,
            "timeout_seconds": number,
            "retry_count": number,
            "retry_delay_seconds": number,
            "depends_on": array,
            "cleanup_commands": array,
            "validation_commands": array
        }
    ]
}""",
                "user": "Please decompose the following task into primitive subtasks: {task}"
            },
            "validate": {
                "system": """You are a task validation expert. Analyze the given task decomposition and identify any issues or improvements needed.

Your response must be a valid JSON object with the following structure:
{
    "is_valid": boolean,
    "issues": array,
    "suggested_fixes": array
}""",
                "user": "Please validate the following task decomposition: {task}"
            },
            "fix": {
                "system": """You are a task repair expert. Given a task with issues, suggest fixes to make it valid and executable.

Your response must be a valid JSON object with the following structure:
{
    "fixed_task": object,
    "changes_made": array,
    "validation_status": {
        "is_valid": boolean,
        "remaining_issues": array
    }
}""",
                "user": "Please fix the following task: {task}"
            },
            "validate_task": {
                "system": """You are a task validation expert. Analyze the given task and verify it meets all requirements and constraints.

Your response must be a valid JSON object with the following structure:
{
    "is_valid": boolean,
    "issues": array,
    "suggested_fixes": array,
    "validation_details": {
        "structure_valid": boolean,
        "dependencies_valid": boolean,
        "environment_valid": boolean,
        "command_valid": boolean
    }
}""",
                "user": "Please validate the following task: {task}"
            },
            "validate_dag": {
                "system": """You are a DAG validation expert. Analyze the given task graph and verify it is acyclic and properly ordered.

Your response must be a valid JSON object with the following structure:
{
    "is_valid": boolean,
    "issues": array,
    "cycles_detected": array,
    "invalid_dependencies": array,
    "suggested_fixes": array
}""",
                "user": "Please validate the following task DAG: {task}"
            },
            "task_analysis": {
                "system": """You are a task analysis expert. Analyze the given task and provide insights about its structure, dependencies, and potential issues.

Your response must be a valid JSON object with the following structure:
{
    "task_type": string,
    "complexity": string,
    "dependencies": array,
    "potential_issues": array,
    "optimization_suggestions": array
}""",
                "user": "Please analyze the following task: {task}"
            },
            "pattern_analysis": {
                "system": """You are a pattern analysis expert. Analyze the given tasks and identify common patterns, anti-patterns, and potential optimizations.

Your response must be a valid JSON object with the following structure:
{
    "common_patterns": array,
    "anti_patterns": array,
    "optimization_opportunities": array,
    "global_recommendations": array
}""",
                "user": "Please analyze patterns in the following tasks: {tasks}"
            },
            "feedback": {
                "system": """You are a feedback generation expert. Analyze the task execution results and provide detailed, actionable feedback.

Your response must be a valid JSON object with the following structure:
{
    "success": boolean,
    "issues": array,
    "recommendations": array,
    "priority_fixes": array
}""",
                "user": "Please generate feedback for the following execution results: {task}"
            },
            "analyze_tasks": {
                "system": """You are a task analysis expert. Analyze multiple tasks and their relationships to identify dependencies and execution order.

Your response must be a valid JSON object with the following structure:
{
    "execution_order": array,
    "dependencies": object,
    "potential_bottlenecks": array,
    "optimization_suggestions": array
}""",
                "user": "Please analyze the following tasks: {tasks}"
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

    async def get_llm_response(self, prompt_type: str, **kwargs) -> Dict[str, Any]:
        """Get response from LLM for a given prompt type"""
        try:
            self.logger.debug(f"Starting LLM request for {prompt_type}")

            if prompt_type not in self.prompts:
                raise LLMError(f"Unknown prompt type: {prompt_type}")

            prompt = self.prompts[prompt_type]
            
            # Extract task from kwargs based on prompt type
            task_data = None
            if "task" in kwargs:
                task_data = kwargs["task"]
            elif "tasks" in kwargs:
                task_data = kwargs["tasks"]
            elif "content" in kwargs:
                task_data = kwargs["content"]
            
            # Format prompt with task data
            formatted_prompt = prompt["user"].format(task=task_data)

            self.logger.debug("Formatted prompt for " + prompt_type)

            start_time = datetime.now()

            # Combine system prompts
            system_content = prompt["system"] + "\n\n" + "IMPORTANT: Your response MUST be a valid JSON object matching the schema specified. Do not include any text outside the JSON object."
            
            # Create message with combined system prompt and user content
            messages = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ]

            # Get response from LLM
            async with self._semaphore:
                response = await self.model.agenerate([messages])
                elapsed = datetime.now() - start_time
                self.logger.info(f"LLM request completed in {elapsed.total_seconds()}s")

                # Parse response
                parsed_response = await self.parse_llm_response(response.generations[0][0].text)
                return parsed_response

        except Exception as e:
            self.logger.error(f"Error getting LLM response: {str(e)}")
            raise LLMError(f"Failed to get LLM response: {str(e)}")

    async def parse_llm_response(self, response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        try:
            # If response is already a dict, return it
            if isinstance(response, dict):
                return response
                
            # Try to parse as JSON first
            try:
                parsed = json.loads(response)
                return parsed
            except json.JSONDecodeError as e:
                self.logger.debug(f"Direct JSON parse failed: {e}")

            # Try to extract JSON from markdown code blocks
            json_matches = re.finditer(r'```(?:json)?\s*({[^`]*})\s*```', response)
            for match in json_matches:
                try:
                    parsed = json.loads(match.group(1))
                    return parsed
                except json.JSONDecodeError as e:
                    self.logger.debug(f"Code block JSON parse failed: {e}")
                    continue

            # Try to find any JSON-like structure
            potential_json = re.search(r'({[\s\S]*})', response)
            if potential_json:
                try:
                    # Clean up the text before parsing
                    cleaned_json = re.sub(r'(?m)^\s*//.*\n?', '', potential_json.group(1))  # Remove comments
                    cleaned_json = re.sub(r',\s*}', '}', cleaned_json)  # Remove trailing commas
                    cleaned_json = re.sub(r',\s*]', ']', cleaned_json)  # Remove trailing commas in arrays
                    parsed = json.loads(cleaned_json)
                    return parsed
                except json.JSONDecodeError as e:
                    self.logger.debug(f"Cleaned JSON parse failed: {e}")

            # If we can't parse JSON, log the response and raise an error
            self.logger.error(f"Could not parse response as JSON: {response[:200]}...")
            raise LLMError(
                "LLM response is not in the required JSON format. "
                "Please ensure the response follows the schema specified in the prompt."
            )

        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {str(e)}")
            raise LLMError(f"Failed to parse LLM response: {str(e)}")

    async def log_action(self, action: str, details: any):
        """Log an action with details"""
        self.logger.info(f"Action: {action}, Details: {details}")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources"""
        if self._client:
            await self._client.aclose()