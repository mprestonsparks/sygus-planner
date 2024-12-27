# tests/test_integration.py

import pytest
import asyncio
from datetime import datetime
from src.agents.orchestrator import OrchestratorAgent
from src.core.data_structures import PrimitiveTask, ValidationResult

class TestDAGGenerationSystem:
    @pytest.fixture
    def system(self):
        return OrchestratorAgent()

    @pytest.mark.asyncio
    async def test_basic_dag_generation(self, system):
        input_tasks = [
            {
                "id": "simple_task",
                "type": "shell_command",
                "description": "Echo hello world",
                "command": "echo 'hello world'"
            }
        ]
        
        result = await system.generate_dag(input_tasks)
        assert result is not None
        assert 'primitive_tasks' in result
        assert len(result['primitive_tasks']) > 0

    @pytest.mark.asyncio
    async def test_complex_dag_generation(self, system):
        input_tasks = [
            {
                "id": "setup_env",
                "type": "environment_setup",
                "description": "Setup Python virtual environment",
                "requirements": {
                    "python_version": "3.8"
                }
            }
        ]
        
        result = await system.generate_dag(input_tasks)
        assert result is not None
        assert result['metadata']['validation_passed']
        assert result['metadata']['error_count'] == 0

    @pytest.mark.asyncio
    async def test_error_handling(self, system):
        input_tasks = [
            {
                "id": "invalid_task",
                "type": "unknown_type",
                "description": "This should fail"
            }
        ]
        
        with pytest.raises(Exception):
            await system.generate_dag(input_tasks)

    @pytest.mark.asyncio
    async def test_llm_integration(self, system):
        input_tasks = [
            {
                "id": "test_task",
                "type": "shell_command",
                "description": "Test command",
                "command": "test -f file.txt"
            }
        ]
        
        result = await system.generate_dag(input_tasks)
        assert 'llm_insights' in result
        assert len(result['llm_insights']) > 0

def run_tests():
    pytest.main([__file__])

if __name__ == "__main__":
    run_tests()