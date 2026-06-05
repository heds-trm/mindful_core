# Installation

## Prerequisites

* Python 3.10+ (or the version required by the project)
* [Poetry](https://python-poetry.org/docs/)

Verify your installation:

```bash
python --version
poetry --version
```

## Clone the Repository

```bash
git clone https://github.com/heds-trm/mindful_core.git
cd mindful_core
```

## Install Dependencies

Install all project dependencies defined in `pyproject.toml`:

```bash
poetry install
```

## Running Commands

Run Python scripts within the project's virtual environment using:

```bash
poetry run python <script>.py
```

For example:

```bash
poetry run python main.py
```

## Recommended: Activate the Virtual Environment

To work directly inside the virtual environment:

```bash
source $(poetry env info --path)/bin/activate
```

On Windows (PowerShell):

```powershell
& "$(poetry env info --path)\Scripts\Activate.ps1"
```

Alternatively, install the Poetry shell plugin:

```bash
poetry self add poetry-plugin-shell
poetry shell
```

If your poetry version is < 2.0, the poetry shell command is available by default.

## Updating Dependencies

```bash
poetry update
```
