# tests/test_agents/test_orchestrator.py

import pytest
import asyncio
from datetime import datetime
from src.agents.orchestrator import OrchestratorAgent
from src.core.data_structures import PrimitiveTask, ValidationResult, State, ReActStep

class TestOrchestratorAgent:
   @pytest.fixture
   def orchestrator(self):
       return OrchestratorAgent()

   @pytest.fixture
   def sample_tasks(self):
       return [
           {
               "id": "test_task",
               "type": "shell_command",
               "description": "Test command execution",
               "command": "echo 'test'"
           }
       ]

   @pytest.mark.asyncio
   async def test_orchestrate(self, orchestrator, sample_tasks):
       result = await orchestrator.orchestrate(sample_tasks)
       assert isinstance(result, dict)
       assert 'dag_id' in result
       assert 'primitive_tasks' in result
       assert 'validation_result' in result
       assert 'error_analysis' in result
       assert 'feedback' in result
       assert 'metadata' in result

   @pytest.mark.asyncio
   async def test_process_task_with_llm(self, orchestrator, sample_tasks):
       react_step = await orchestrator.process_task_with_llm(sample_tasks[0])
       assert isinstance(react_step, ReActStep)
       assert react_step.thought
       assert react_step.action
       assert react_step.observation
       assert react_step.reflection

   @pytest.mark.asyncio
   async def test_get_llm_thought(self, orchestrator, sample_tasks):
       thought = await orchestrator.get_llm_thought(sample_tasks[0], None)
       assert isinstance(thought, str)
       assert thought

   @pytest.mark.asyncio
   async def test_get_llm_observation(self, orchestrator, sample_tasks):
       primitive_task = PrimitiveTask(
           task_id="test_1",
           type="shell_command",
           command="echo 'test'",
           working_directory="/tmp",
           environment={},
           error_pattern="",
           success_pattern="",
           timeout_seconds=30,
           retry_count=3,
           retry_delay_seconds=5,
           depends_on=[],
           cleanup_commands=[],
           validation_commands=[]
       )
       validation_result = ValidationResult(
           is_valid=True,
           issues=[],
           suggested_fixes=[],
           validation_time=datetime.utcnow(),
           validator_id="test"
       )
       observation = await orchestrator.get_llm_observation(
           sample_tasks[0],
           [primitive_task],
           [validation_result]
       )
       assert isinstance(observation, str)
       assert observation

   @pytest.mark.asyncio
   async def test_validate_primitive(self, orchestrator):
       primitive_task = {
           "task_id": "test_1",
           "type": "shell_command",
           "command": "echo 'test'",
           "working_directory": "/tmp",
           "environment": {"PATH": "/usr/bin"},
           "error_pattern": "error",
           "success_pattern": "success",
           "timeout_seconds": 30,
           "retry_count": 3,
           "retry_delay_seconds": 5,
           "depends_on": [],
           "cleanup_commands": [],
           "validation_commands": []
       }
       is_valid = await orchestrator.validate_primitive(primitive_task)
       assert isinstance(is_valid, bool)

   @pytest.mark.asyncio
   async def test_validate_dag_with_llm(self, orchestrator):
       primitive_task = PrimitiveTask(
           task_id="test_1",
           type="shell_command",
           command="echo 'test'",
           working_directory="/tmp",
           environment={},
           error_pattern="",
           success_pattern="",
           timeout_seconds=30,
           retry_count=3,
           retry_delay_seconds=5,
           depends_on=[],
           cleanup_commands=[],
           validation_commands=[]
       )
       result = await orchestrator.validate_dag_with_llm([primitive_task])
       assert isinstance(result, ValidationResult)
       assert hasattr(result, 'is_valid')
       assert hasattr(result, 'issues')
       assert hasattr(result, 'suggested_fixes')

   @pytest.mark.asyncio
   async def test_generate_enhanced_feedback(self, orchestrator):
       validation_result = ValidationResult(
           is_valid=True,
           issues=[],
           suggested_fixes=[],
           validation_time=datetime.utcnow(),
           validator_id="test"
       )
       error_patterns = {}
       initial_analysis = {"insights": []}
       
       feedback = await orchestrator.generate_enhanced_feedback(
           validation_result,
           error_patterns,
           initial_analysis
       )
       assert isinstance(feedback, dict)