# examples/basic_usage.py

import asyncio
import logging
from sygus_planner import DAGGenerationSystem

async def main():
    config = {
        'llm': {
            'model_name': 'claude-3-opus-20240229',
        },
        'logging': {
            'level': logging.INFO,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }
    
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
    
    try:
        system = DAGGenerationSystem()
        result = await system.generate_dag(
            input_tasks=input_tasks,
            output_file="generated_dag.json",
            config=config
        )
        
        print("\nDAG Generation Summary:")
        print(f"Total primitive tasks: {result['metadata']['total_tasks']}")
        print(f"Validation passed: {result['metadata']['validation_passed']}")
        print(f"Total errors detected: {result['metadata']['error_count']}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())