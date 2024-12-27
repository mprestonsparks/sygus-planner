# setup.py

from setuptools import setup, find_packages

setup(
   name="sygus-planner",
   version="0.1.0",
   packages=find_packages(),
   install_requires=[
       "langchain>=0.1.0",
       "anthropic>=0.3.0",
       "z3-solver>=4.8.0",
       "dataclasses-json>=0.5.0",
       "networkx>=2.5",
       "pytest>=6.0.0",
       "pytest-asyncio>=0.14.0",
       "pytest-cov>=2.10.0",
       "black>=20.8b1",
       "isort>=5.6.0",
       "flake8>=3.8.0",
   ],
   extras_require={
       "dev": [
           "black",
           "isort",
           "flake8",
           "pytest",
           "pytest-asyncio",
           "pytest-cov",
       ]
   },
   python_requires=">=3.8",
   author="Your Name",
   author_email="your.email@example.com",
   description="A Task Decomposition Engine using SyGuS and LLM reasoning",
   long_description=open("README.md").read(),
   long_description_content_type="text/markdown",
   url="https://github.com/yourusername/sygus-planner",
   classifiers=[
       "Development Status :: 3 - Alpha",
       "Intended Audience :: Developers",
       "License :: OSI Approved :: MIT License",
       "Programming Language :: Python :: 3.8",
       "Programming Language :: Python :: 3.9",
       "Programming Language :: Python :: 3.10",
   ],
)