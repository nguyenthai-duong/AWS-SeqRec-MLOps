SHELL = /bin/bash

.PHONY: style test notebook precommit clean

style:
	uv run black src/ data_pipeline_aws/
	uv run isort src/ data_pipeline_aws/
	uv run flake8 src/ data_pipeline_aws/
	uv run nbqa black notebooks/
	uv run nbqa isort notebooks/
	uv run nbqa flake8 notebooks/
	uv run mypy src/ data_pipeline_aws/

test:
	uv run pytest

notebook:
	uv run jupyter lab

precommit:
	uv run pre-commit install

