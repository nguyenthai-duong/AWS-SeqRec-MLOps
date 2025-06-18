SHELL = /bin/bash

.PHONY: style test notebook precommit clean

style:
	uv run black src/
	uv run isort src/
	uv run flake8 src/
	uv run nbqa black notebooks/
	uv run nbqa isort notebooks/
	uv run nbqa flake8 notebooks/
	uv run mypy src/

test:
	uv run pytest

notebook:
	uv run jupyter lab

precommit:
	uv run pre-commit install

