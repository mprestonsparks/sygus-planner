# src/core/exceptions.py

class DAGValidationError(Exception):
    """Raised when DAG validation fails"""
    pass

class TaskDecompositionError(Exception):
    """Raised when task decomposition fails"""
    pass

class StateUpdateError(Exception):
    """Raised when state update fails"""
    pass

class LLMError(Exception):
    """Raised when LLM interaction fails"""
    pass