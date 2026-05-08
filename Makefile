# Makefile for Time-Series Classification Project

.PHONY: help install test clean run notebooks

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make clean         - Clean up cache files"
	@echo "  make notebooks     - Run Jupyter notebooks"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with flake8"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

notebooks:
	jupyter notebook notebooks/

format:
	black src/ tests/

lint:
	flake8 src/ tests/ --max-line-length=100

.DEFAULT_GOAL := help
