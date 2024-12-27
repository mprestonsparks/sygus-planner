# src/state/manager.py

import json
from datetime import datetime
from typing import Dict, Optional
from ..core.data_structures import State
from ..core.exceptions import StateUpdateError

class StateManager:
   def __init__(self):
       self.states: Dict[str, State] = {}
       self.current_state_id: Optional[str] = None

   def create_new_state(self) -> State:
       state = State.create_initial()
       self.states[state.id] = state
       self.current_state_id = state.id
       return state

   def get_current_state(self) -> State:
       if not self.current_state_id:
           raise StateUpdateError("No current state exists")
       return self.states[self.current_state_id]

   def update_state(self, state: State):
       state.last_updated = datetime.utcnow()
       self.states[state.id] = state

   def save_state(self, filepath: str):
       if not self.current_state_id:
           raise StateUpdateError("No current state to save")
       
       state = self.get_current_state()
       state_dict = asdict(state)
       state_dict['completed_tasks'] = list(state_dict['completed_tasks'])
       state_dict['pending_tasks'] = list(state_dict['pending_tasks'])
       
       with open(filepath, 'w') as f:
           json.dump(state_dict, f, default=str)

   def load_state(self, filepath: str):
       with open(filepath, 'r') as f:
           state_dict = json.load(f)
           
       state_dict['completed_tasks'] = set(state_dict['completed_tasks'])
       state_dict['pending_tasks'] = set(state_dict['pending_tasks'])
       state_dict['creation_time'] = datetime.fromisoformat(state_dict['creation_time'])
       state_dict['last_updated'] = datetime.fromisoformat(state_dict['last_updated'])
       
       state = State(**state_dict)
       self.states[state.id] = state
       self.current_state_id = state.id