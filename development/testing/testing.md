# Testing

This document describes the testing strategy and procedures for `synth-pdb`.

## Testing Framework

We use [pytest](https://docs.pytest.org/en/stable/) as our primary testing framework. Our test suite is designed to ensure the reliability and correctness of protein structure generation, biophysical calculations, and NMR predictions.

## Test Organization

The `tests/` directory is organized into several subdirectories based on the type of tests:

-   `unit/`: Tests for individual functions and classes in isolation.
-   `functional/`: Tests for end-to-end functionality of the CLI and core modules.
-   `integration/`: Tests that verify the interaction between different modules.
-   `geometry/`: Specialized tests for coordinate transformations and NeRF calculations.

## Running Tests

To run the full test suite, execute the following command from the project root:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_generator.py
```

To run tests with a specific marker (if available):

```bash
pytest -m "not slow"
```

## Coverage

We use `pytest-cov` to measure code coverage. Aim for at least 80% coverage on all core modules.

To run tests with a coverage report:

```bash
pytest --cov=synth_pdb --cov-report=term-missing
```

This will generate a summary in the terminal showing the coverage percentage for each file.

## Continuous Integration (CI)

Our CI pipeline is managed through GitHub Actions. Every push and pull request to the `main` branch triggers a suite of automated tests across different operating systems (Linux, macOS) and Python versions (3.10, 3.11, 3.12).

The CI workflow (`.github/workflows/test.yml`) includes:
-   **Static Analysis:** Linting with `ruff` and type checking with `mypy`.
-   **Test Execution:** Running the full `pytest` suite.
-   **Coverage Upload:** Uploading coverage reports to Codecov for tracking.

## Writing New Tests

When adding new features or fixing bugs, please include corresponding tests:
1.  **For Bugs:** Create a regression test that fails without the fix and passes with it.
2.  **For Features:** Add unit tests for the new functionality and integration tests for its usage within the system.

Please ensure all tests are well-documented and follow the existing naming conventions (`test_*.py`).
