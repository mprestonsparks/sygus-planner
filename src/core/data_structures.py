# src/core/data_structures.py

# src/core/data_structures.py
from dataclasses import dataclass
from typing import Any, List, Dict, Set, Optional, Union
from datetime import datetime

@dataclass
class PrimitiveTask:
    task_id: str
    type: str  # shell_command, api_call, file_operation
    command: str
    working_directory: str
    environment: Dict[str, str]
    error_pattern: str
    success_pattern: str
    timeout_seconds: int
    retry_count: int
    retry_delay_seconds: int
    depends_on: List[str]
    cleanup_commands: List[str]
    validation_commands: List[str]

@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[str]
    suggested_fixes: List[str]
    validation_time: datetime
    validator_id: str

@dataclass
class State:
    id: str
    current_phase: str
    completed_tasks: Set[str]
    pending_tasks: Set[str]
    validation_status: Dict[str, ValidationResult]
    error_states: List[str]
    creation_time: datetime
    last_updated: datetime
    
    @classmethod
    def create_initial(cls) -> 'State':
        now = datetime.utcnow()
        return cls(
            id=str(uuid.uuid4()),
            current_phase="initialization",
            completed_tasks=set(),
            pending_tasks=set(),
            validation_status={},
            error_states=[],
            creation_time=now,
            last_updated=now
        )

@dataclass
class LLMResponse:
    content: str
    model: str
    finish_reason: str
    raw_response: Any

@dataclass
class ReActStep:
    id: str
    thought: str
    action: str
    observation: str
    reflection: str
    timestamp: datetime
    agent_id: str
    
    @classmethod
    def create(cls, agent_id: str) -> 'ReActStep':
        return cls(
            id=str(uuid.uuid4()),
            thought="",
            action="",
            observation="",
            reflection="",
            timestamp=datetime.utcnow(),
            agent_id=agent_id
        )