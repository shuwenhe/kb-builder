.PHONY: help venv install test example-build example-query clean

help:
	@echo "KB Builder - Knowledge Base Builder"
	@echo ""
	@echo "Targets:"
	@echo "  venv          - Create Python virtual environment"
	@echo "  install       - Install dependencies"
	@echo "  test          - Run tests"
	@echo "  example-build - Run example build script"
	@echo "  example-query - Run example query script"
	@echo "  clean         - Remove generated files"

venv:
	python3 -m venv .venv
	@echo "Virtual environment created. Activate with: source .venv/bin/activate"

install:
	pip install -e .


test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v

example-build:
	python example_build.py

example-query:
	python example_load.py

clean:
	rm -rf kb/ .pytest_cache/ __pycache__/ kb_builder/__pycache__/ kb_builder.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
