"""Setup configuration for KB Builder."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="kb-builder",
    version="0.1.0",
    author="Your Name",
    description="Knowledge Base Builder for RAG Applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "faiss-cpu>=1.8.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pydantic>=2.0.0",
        "docx-parser>=0.1.0",
        "langchain>=0.3.0",
        "langchain-community>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
