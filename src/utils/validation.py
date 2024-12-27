# src/utils/validation.py

from pathlib import Path
from typing import Callable, Dict
from ..core.data_structures import PrimitiveTask

def validate_primitive_task(task: PrimitiveTask) -> bool:
    required_fields = {
        'task_id': bool,
        'type': lambda x: x in {'shell_command', 'api_call', 'file_operation'},
        'command': bool,
        'working_directory': lambda x: Path(x).is_absolute(),
        'environment': lambda x: isinstance(x, dict),
        'error_pattern': bool,
        'success_pattern': bool,
        'timeout_seconds': lambda x: isinstance(x, int) and x > 0,
        'retry_count': lambda x: isinstance(x, int) and x >= 0,
        'retry_delay_seconds': lambda x: isinstance(x, int) and x >= 0,
        'depends_on': list,
        'cleanup_commands': list,
        'validation_commands': list
    }
    
    try:
        task_dict = asdict(task)
        return all(
            validator(task_dict[field]) if callable(validator) else task_dict[field]
            for field, validator in required_fields.items()
        )
    except Exception as e:
        return False