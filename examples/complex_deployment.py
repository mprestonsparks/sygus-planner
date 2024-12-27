# examples/complex_deployment.py

import asyncio
import logging
from sygus_planner import DAGGenerationSystem

async def main():
   config = {
       'llm': {'model_name': 'claude-3-opus-20240229'},
       'logging': {
           'level': logging.INFO,
           'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
       }
   }
   
   input_tasks = [
       {
           "id": "setup_k8s",
           "type": "infrastructure_setup",
           "description": "Setup Kubernetes cluster and necessary resources",
           "requirements": {
               "cluster_size": 3,
               "region": "us-west-2",
               "k8s_version": "1.24",
               "resources": {
                   "databases": ["postgresql"],
                   "monitoring": ["prometheus", "grafana"],
                   "ingress": "nginx"
               }
           }
       },
       {
           "id": "deploy_app",
           "type": "application_deployment",
           "description": "Deploy microservices application",
           "requirements": {
               "services": [
                   {
                       "name": "auth-service",
                       "replicas": 3,
                       "resources": {"cpu": "500m", "memory": "512Mi"},
                       "dependencies": ["postgresql"]
                   },
                   {
                       "name": "api-gateway",
                       "replicas": 2,
                       "resources": {"cpu": "1000m", "memory": "1Gi"},
                       "dependencies": ["auth-service"]
                   }
               ],
               "configs": ["config-maps", "secrets"],
               "monitoring": True
           }
       }
   ]
   
   try:
       system = DAGGenerationSystem()
       result = await system.generate_dag(
           input_tasks=input_tasks,
           output_file="k8s_deployment_dag.json",
           config=config
       )
       
       print("\nDeployment DAG Generation Summary:")
       print(f"Total primitive tasks: {result['metadata']['total_tasks']}")
       print(f"Validation passed: {result['metadata']['validation_passed']}")
       print(f"Total errors detected: {result['metadata']['error_count']}")
       
       print("\nLLM Insights:")
       for insight in result['llm_insights']:
           print(f"- {insight}")
       
       if not result['validation_result']['is_valid']:
           print("\nValidation Issues:")
           for issue in result['validation_result']['issues']:
               print(f"- {issue}")
       
       print("\nRecommendations:")
       for rec in result['feedback'].get('recommendations', []):
           print(f"- {rec}")
           
   except Exception as e:
       print(f"Error: {e}")
       raise

if __name__ == "__main__":
   asyncio.run(main())