# src/llm/manager.py

from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from typing import Dict, Any, Optional
import json
import logging
from pathlib import Path
from ..core.data_structures import LLMResponse
from ..core.exceptions import LLMError

class LLMManager:
    def __init__(self, model_name: str = "claude-3-opus-20240229"):
        self.llm = ChatAnthropic(model=model_name)
        self.logger = logging.getLogger("LLMManager")
        self.prompts = self.load_prompt_templates()

    def load_prompt_templates(self) -> Dict[str, str]:
        prompt_dir = Path(__file__).parent / "prompts"
        templates = {}
        for template_file in prompt_dir.glob("*.txt"):
            templates[template_file.stem] = template_file.read_text()
        return templates

    async def get_llm_response(self, 
                              prompt_type: str, 
                              content: Any,
                              additional_context: Optional[Dict] = None) -> LLMResponse:
        if prompt_type not in self.prompts:
            raise LLMError(f"Unknown prompt type: {prompt_type}")
        
        system_prompt = self.prompts[prompt_type]
        if additional_context:
            system_prompt += f"\nAdditional Context: {json.dumps(additional_context)}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(content))
        ]
        
        try:
            response = await self.llm.agenerate([messages])
            
            return LLMResponse(
                content=response.generations[0][0].text,
                model=response.llm_output.get("model_name", "unknown"),
                finish_reason=response.generations[0][0].finish_reason,
                raw_response=response
            )
            
        except Exception as e:
            self.logger.error(f"LLM request failed: {e}")
            raise LLMError(f"Failed to get LLM response: {e}")

    async def parse_llm_response(self, response: LLMResponse) -> Dict:
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            raise LLMError("LLM response was not valid JSON")