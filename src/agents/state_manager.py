# src/agents/state_manager.py

"""State manager for tracking task execution state."""
import uuid
from datetime import datetime
from typing import Dict, Optional, Set

from ..core.data_structures import State, ValidationResult

class StateManager:
    def __init__(self):
        self.states: Dict[str, State] = {}

    def create_new_state(self) -> State:
        """Create a new state with a unique ID"""
        state_id = str(uuid.uuid4())
        state = State(
            id=state_id,
            current_phase="initialization",
            completed_tasks=set(),
            pending_tasks=set(),
            validation_status={},
            error_states=[],
            creation_time=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        self.states[state_id] = state
        return state

    def get_state(self, state_id: str) -> Optional[State]:
        """Get state by ID"""
        return self.states.get(state_id)

    def update_state(self, state: State) -> None:
        """Update an existing state"""
        state.last_updated = datetime.utcnow()
        self.states[state.id] = state

    def delete_state(self, state_id: str) -> None:
        """Delete a state by ID"""
        if state_id in self.states:
            del self.states[state_id]
