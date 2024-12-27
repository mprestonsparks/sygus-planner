# src/agents/state_manager.py

from ..core.data_structures import State
from typing import Dict, Any
import uuid
from datetime import datetime

class StateManager:
    def __init__(self):
        self.states: Dict[str, State] = {}

    def create_new_state(self) -> State:
        """Create a new state with a unique ID"""
        state_id = str(uuid.uuid4())
        state = State(
            state_id=state_id,
            creation_time=datetime.utcnow(),
            last_update=datetime.utcnow(),
            data={}
        )
        self.states[state_id] = state
        return state

    def update_state(self, state: State, data: Dict[str, Any] = None) -> State:
        """Update an existing state with new data"""
        if data:
            state.data.update(data)
        state.last_update = datetime.utcnow()
        self.states[state.state_id] = state
        return state

    def get_state(self, state_id: str) -> State:
        """Get a state by its ID"""
        return self.states.get(state_id)

    def list_states(self) -> Dict[str, State]:
        """Get all states"""
        return self.states

    def delete_state(self, state_id: str) -> bool:
        """Delete a state by its ID"""
        if state_id in self.states:
            del self.states[state_id]
            return True
        return False
