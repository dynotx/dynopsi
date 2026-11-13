# keep basic commands at the top


# default recipe if none are passed: lint-all
default: lint-all

# type check with mypy
mypy:
    uv run mypy --sqlite-cache

# format with ruff
format:
    uv run ruff format

# lint with ruff
lint:
    uv run ruff check --fix

# run all linting (format, lint, type check)
lint-all: format lint mypy

# lint like CI, fail if there are any changes
ci-lint: lint-all

test-unit:
    uv run pytest tests/unit

test-e2e:
    uv run bash tests/e2e/cli.sh --clean

test-all: test-unit test-e2e

# default test is only unit tests
test: test-unit

# sync python packages to project specs, removes non-project packages
sync:
    uv sync --locked

# sync python packages to project specs, keeps non-project packages
sync-inexact:
    uv sync --locked --inexact

# update to latest main and sync python packages
up:
    git checkout main
    git pull --prune
    uv sync --locked

# bootstrap the python environment
bootstrap:
    #!/bin/bash
    set -euo pipefail

    # vars set via just
    os={{ os() }}
    venv_exists={{ path_exists(".venv") }}

    echo 'Bootstrapping...'

    echo 'Installing Python specified by .python-version'
    uv python install

    if $venv_exists; then
        echo 'Virtual environment .venv exists, skipping creation'
    else
        echo 'Creating Python virtual environment .venv'
        uv venv --seed
    fi

    echo 'Installing Python packages'
    uv sync --locked

    echo 'Bootstrapping complete'
