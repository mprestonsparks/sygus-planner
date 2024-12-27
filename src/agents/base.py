# src/agents/base.py

import logging
import uuid
from datetime import datetime
from typing import Any

class BaseAgent:
   def __init__(self, agent_id: str, agent_type: str):
       self.agent_id = agent_id
       self.agent_type = agent_type
       self.logger = logging.getLogger(f"{agent_type}_{agent_id}")

   async def log_action(self, action: str, details: Any = None):
       self.logger.info(f"Agent {self.agent_id} performing {action}: {details}")

   async def validate_action(self, action: str, params: dict) -> bool:
       """Validate an action before performing it"""
       return True  # Override in subclasses

   async def handle_error(self, error: Exception, context: dict) -> None:
       """Handle errors during agent operation"""
       self.logger.error(f"Error in {self.agent_type}: {error}")
       self.logger.debug(f"Error context: {context}")