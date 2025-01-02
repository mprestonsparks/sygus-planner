import pytest
import asyncio
import logging
from typing import Generator
import pytest_asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set default fixture scope for async fixtures
pytest.register_assert_rewrite('tests.helpers')

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    logger.info("Setting up test event loop")
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    logger.info("Cleaning up test event loop")
    loop.close()

@pytest_asyncio.fixture(scope="function")
async def async_client():
    """Async client fixture that ensures proper cleanup."""
    logger.info("Setting up async client")
    start_time = datetime.now()
    yield
    # Cleanup any remaining connections
    await asyncio.sleep(0)  # Allow pending tasks to complete
    duration = datetime.now() - start_time
    logger.info(f"Async client fixture duration: {duration.total_seconds()}s")

@pytest_asyncio.fixture
async def llm_manager():
    """Provide a shared LLM manager instance for tests."""
    from src.llm.manager import LLMManager
    logger.info("Creating LLM manager instance")
    async with LLMManager() as manager:
        yield manager
    logger.info("Cleaned up LLM manager instance")

@pytest_asyncio.fixture
async def decomposer(llm_manager):
    """Provide a decomposer agent with managed LLM instance."""
    from src.agents.decomposer import DecomposerAgent
    logger.info("Creating decomposer agent")
    agent = DecomposerAgent(llm_manager=llm_manager)
    yield agent
    logger.info("Cleaned up decomposer agent")

@pytest_asyncio.fixture
async def validator(llm_manager):
    """Provide a validator agent with managed LLM instance."""
    from src.agents.validator import ValidatorAgent
    logger.info("Creating validator agent")
    agent = ValidatorAgent(llm_manager=llm_manager)
    yield agent
    logger.info("Cleaned up validator agent")

@pytest_asyncio.fixture
async def orchestrator(llm_manager, decomposer, validator):
    """Provide an orchestrator with managed dependencies."""
    from src.agents.orchestrator import OrchestratorAgent
    logger.info("Creating orchestrator agent")
    agent = OrchestratorAgent(
        llm_manager=llm_manager,
        decomposer=decomposer,
        validator=validator
    )
    yield agent
    logger.info("Cleaned up orchestrator agent")

# Configure pytest-asyncio
def pytest_configure(config):
    config.option.asyncio_mode = "auto"
    config.option.asyncio_default_fixture_loop_scope = "function"
    config.addinivalue_line(
        "markers", "timeout: mark test to timeout after X seconds"
    )

# Add test timeout
def pytest_addoption(parser):
    parser.addoption("--test-timeout", action="store", default=30,
                    type=int, help="Timeout in seconds for each test")

@pytest.fixture(autouse=True)
def test_timeout(request):
    """Add timeout to all tests"""
    timeout = request.config.getoption("--test-timeout", default=30)
    request.node.add_marker(pytest.mark.timeout(timeout))
