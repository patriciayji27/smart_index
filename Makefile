.PHONY: install test lint clean

install:
	pip install -e ".[dev,models]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/smart_index/

format:
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache dist build *.egg-info
