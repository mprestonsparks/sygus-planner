# sygus-planner
A Task Decomposition Engine: Automated conversion of high-level software development requirements into primitive operations using SyGuS (Syntax-Guided Synthesis), formal methods, and LLM reasoning.

## Directory Structure
``` plaintext
sygus-planner/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data_structures.py      # Core data classes and types
│   │   └── exceptions.py           # Custom exception definitions
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                 # Base agent class
│   │   ├── decomposer.py          # Task decomposition agent
│   │   ├── validator.py           # Task validation agent
│   │   ├── error_detector.py      # Error detection agent
│   │   └── orchestrator.py        # Main orchestration agent
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── manager.py             # LLM interaction management
│   │   └── prompts/               # LLM prompt templates
│   │       ├── __init__.py
│   │       ├── decompose.txt
│   │       ├── validate.txt
│   │       └── error_detect.txt
│   ├── state/
│   │   ├── __init__.py
│   │   └── manager.py             # State management
│   └── utils/
│       ├── __init__.py
│       └── validation.py          # Utility validation functions
├── tests/
│   ├── __init__.py
│   ├── test_agents/
│   │   ├── __init__.py
│   │   ├── test_decomposer.py
│   │   ├── test_validator.py
│   │   └── test_orchestrator.py
│   └── test_integration.py
└── examples/
    ├── __init__.py
    ├── basic_usage.py
    └── complex_deployment.py
```

---

## Installation
``` bash
pip install -r requirements.txt
```


## Usage
``` python
from sygus_planner import DAGGenerationSystem

async def main():
    system = DAGGenerationSystem()
    
    input_tasks = [
        {
            "id": "setup_env",
            "type": "environment_setup",
            "description": "Setup Python virtual environment",
            "requirements": {"python_version": "3.8"}
        }
    ]
    
    result = await system.generate_dag(
        input_tasks=input_tasks,
        output_file="generated_dag.json"
    )
```

## Development
``` bash
# Setup development environment
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8
```