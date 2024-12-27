# tests/test_agents/test_validator.py

import pytest
import asyncio
from datetime import datetime
from src.agents.validator import ValidatorAgent
from src.core.data_structures import PrimitiveTask, ValidationResult

class TestValidatorAgent:
    @pytest.fixture
    def validator(self):
        return ValidatorAgent()

    @pytest.fixture
    def sample_task(self):
        return PrimitiveTask(
            task_id="test_1",
            type="shell_command",
            command="echo 'test'",
            working_directory="/tmp",
            environment={"PATH": "/usr/bin"},
            error_pattern="error|fail",
            success_pattern="success",
            timeout_seconds=30,
            retry_count=3,
            retry_delay_seconds=5,
            depends_on=[],
            cleanup_commands=["rm -f /tmp/test.tmp"],
            validation_commands=["test -f /tmp/test.tmp"]
        )

    @pytest.mark.asyncio
    async def test_validate_task(self, validator, sample_task):
        result = await validator.validate_task(sample_task)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert not result.issues

    @pytest.mark.asyncio
    async def test_validate_task_structure(self, validator, sample_task):
        is_valid, issues, fixes = validator.validate_task_structure(sample_task)
        assert is_valid
        assert not issues
        assert not fixes

    @pytest.mark.asyncio
    async def test_validate_command(self, validator, sample_task):
        is_valid, issues, fixes = validator.validate_command(sample_task)
        assert is_valid
        assert not issues
        assert not fixes

    @pytest.mark.asyncio
    async def test_validate_error_handling(self, validator, sample_task):
        is_valid, issues, fixes = validator.validate_error_handling(sample_task)
        assert is_valid
        assert not issues
        assert not fixes

    @pytest.mark.asyncio
    async def test_validate_shell_command(self, validator, sample_task):
        issues, fixes = validator.validate_shell_command(sample_task)
        assert not issues
        assert not fixes

    @pytest.mark.asyncio
    async def test_validate_api_call(self, validator):
        api_task = PrimitiveTask(
            task_id="api_1",
            type="api_call",
            command="GET https://api.example.com/data",
            working_directory="/",
            environment={"Authorization": "Bearer token"},
            error_pattern="4[0-9]{2}|5[0-9]{2}",
            success_pattern="2[0-9]{2}",
            timeout_seconds=30,
            retry_count=3,
            retry_delay_seconds=5,
            depends_on=[],
            cleanup_commands=[],
            validation_commands=[]
        )
        issues, fixes = validator.validate_api_call(api_task)
        assert not issues
        assert not fixes

    @pytest.mark.asyncio
    async def test_validate_file_operation(self, validator):
        file_task = PrimitiveTask(
            task_id="file_1",
            type="file_operation",
            command="mkdir -p /tmp/test",
            working_directory="/tmp",
            environment={},
            error_pattern="Permission denied|File exists",
            success_pattern="",
            timeout_seconds=5,
            retry_count=3,
            retry_delay_seconds=5,
            depends_on=[],
            cleanup_commands=["rmdir /tmp/test"],
            validation_commands=["test -d /tmp/test"]
        )
        issues, fixes = validator.validate_file_operation(file_task)
        assert not issues
        assert not fixes

    @pytest.mark.asyncio
    async def test_validate_dag(self, validator, sample_task):
        tasks = [sample_task]
        result = await validator.validate_dag(tasks)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert not result.issues