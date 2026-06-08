# Contributing to Mindful-Core

Thank you for your interest in contributing to Mindful-Core.

Mindful-Core is a framework for reproducible AI experimentation in medical imaging. Contributions are welcome in areas such as model development, preprocessing pipelines, evaluation metrics, documentation, and bug fixes.

## Development Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd mindful_core
```

2. Install dependencies using Poetry:

```bash
poetry install
```

3. Run the test or example configurations to verify that the installation is working correctly.

Refer to [`INSTALL.md`](./INSTALL.md).

## Project Structure

The framework is designed to be extensible through configuration files and modular components.

Typical extensions include:

* New neural network architectures
* New preprocessing pipelines
* New training strategies
* New evaluation metrics
* New datasets and data loaders

Whenever possible, new functionality should integrate with the existing configuration system rather than requiring modifications to the experiment engine.

## Coding Guidelines

* Follow existing coding conventions and project structure.
* Document new modules, classes, and functions.
* Keep components modular and reusable.
* Ensure that new functionality remains compatible with the framework's reproducibility mechanisms.

## Reproducibility

Mindful-Core places a strong emphasis on reproducibility. Contributions should:

* Use configuration files whenever possible.
* Preserve experiment serialization and deserialization mechanisms.
* Explicitly document any new dependencies.
* Specify version requirements when appropriate.

## Documentation

Please update the documentation when introducing:

* New models
* New pipelines
* New configuration options
* New datasets

Examples are strongly encouraged.

## Reporting Issues

Bug reports and feature requests can be submitted through the GitHub issue tracker.

When reporting a bug, please include:

* Operating system
* Python version
* Mindful-Core version
* Relevant configuration files
* Error messages and stack traces

## Pull Requests

Pull requests should include:

* A clear description of the change
* Updated documentation when relevant
* Example configuration files if new functionality is introduced

All contributions will be reviewed before integration.
