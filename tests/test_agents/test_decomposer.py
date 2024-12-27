# tests/test_agents/test_decomposer.py

import pytest
import asyncio
from typing import List, Dict
from src.agents.decomposer import DecomposerAgent
from src.core.data_structures import PrimitiveTask
from src.core.exceptions import TaskDecompositionError

class TestDecomposerAgent:
   @pytest.fixture
   def decomposer(self):
       return DecomposerAgent()

   @pytest.fixture
   def sample_task(self):
       return {
           "id": "test_task",
           "type": "shell_command",
           "description": "Echo hello world",
           "command": "echo 'hello world'"
       }

   @pytest.fixture
   def primitive_task(self):
       return {
           "task_id": "primitive_1",
           "type": "shell_command",
           "command": "echo 'hello world'",
           "working_directory": "/tmp",
           "environment": {"PATH": "/usr/bin"},
           "error_pattern": ".*error.*",
           "success_pattern": ".*success.*",
           "timeout_seconds": 30,
           "retry_count": 3,
           "retry_delay_seconds": 5,
           "depends_on": [],
           "cleanup_commands": [],
           "validation_commands": []
       }

   @pytest.mark.asyncio
   async def test_decompose_task_primitive(self, decomposer, primitive_task):
       result = await decomposer.decompose_task(primitive_task)
       assert len(result) == 1
       assert isinstance(result[0], PrimitiveTask)
       assert result[0].task_id == primitive_task["task_id"]

   @pytest.mark.asyncio
   async def test_decompose_task_complex(self, decomposer, sample_task):
       result = await decomposer.decompose_task(sample_task)
       assert len(result) > 0
       assert all(isinstance(task, PrimitiveTask) for task in result)

   @pytest.mark.asyncio
   async def test_validate_decomposition(self, decomposer, primitive_task):
       tasks = [PrimitiveTask(**primitive_task)]
       issues = await decomposer.validate_decomposition(sample_task, tasks)
       assert isinstance(issues, list)

   @pytest.mark.asyncio
   async def test_auto_fix_task(self, decomposer):
       invalid_task = PrimitiveTask(
           task_id="invalid_1",
           type="shell_command",
           command="echo 'test'",
           working_directory="./relative/path",
           environment={},
           error_pattern="",
           success_pattern="",
           timeout_seconds=0,
           retry_count=0,
           retry_delay_seconds=0,
           depends_on=[],
           cleanup_commands=[],
           validation_commands=[]
       )
       fixed_task = await decomposer.auto_fix_task(invalid_task)
       assert fixed_task is not None
       assert fixed_task.working_directory.startswith('/')
       assert 'PATH' in fixed_task.environment
       assert fixed_task.timeout_seconds > 0

   def test_is_primitive_description(self, decomposer, primitive_task):
       assert decomposer.is_primitive_description(primitive_task) is True

   def test_validate_dependencies(self, decomposer):
       tasks = [
           PrimitiveTask(
               task_id="task1",
               type="shell_command",
               command="echo '1'",
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
           ),
           PrimitiveTask(
               task_id="task2",
               type="shell_command",
               command="echo '2'",
               working_directory="/tmp",
               environment={},
               error_pattern="",
               success_pattern="",
               timeout_seconds=30,
               retry_count=3,
               retry_delay_seconds=5,
               depends_on=["task1"],
               cleanup_commands=[],
               validation_commands=[]
           )
       ]
       assert decomposer.validate_dependencies(tasks) is True

   def test_validate_environment(self, decomposer, primitive_task):
       task = {"environment": {"TEST_VAR": "value"}}
       tasks = [PrimitiveTask(**primitive_task)]
       assert decomposer.validate_environment(task, tasks) is False

   def test_validate_file_operations(self, decomposer, primitive_task):
       task = {"working_directory": "/test/dir"}
       tasks = [PrimitiveTask(**primitive_task)]
       assert decomposer.validate_file_operations(task, tasks) is False