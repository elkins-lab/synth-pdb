# Release Process

This document outlines the procedures for releasing new versions of `synth-pdb`.

## Versioning Scheme

We follow [Semantic Versioning (SemVer)](https://semver.org/). Version numbers are formatted as `MAJOR.MINOR.PATCH`:

-   **MAJOR:** Significant changes, potentially breaking API compatibility.
-   **MINOR:** New features added in a backwards-compatible manner.
-   **PATCH:** Backwards-compatible bug fixes and small improvements.

The version number is centrally defined in `pyproject.toml`.

## Release Workflow

A typical release involves the following steps:

1.  **Preparation:**
    -   Ensure all tests are passing.
    -   Update the `CHANGELOG.md` with a summary of changes since the last release.
    -   Increment the version number in `pyproject.toml`.

2.  **Tagging:**
    -   Create a new Git tag for the release:
      ```bash
      git tag -a vX.Y.Z -m "Release vX.Y.Z"
      git push origin vX.Y.Z
      ```

3.  **Documentation Deployment:**
    -   The `Publish Docs` GitHub Actions workflow (`.github/workflows/deploy_docs.yaml`) automatically builds and deploys the documentation to GitHub Pages whenever a change is pushed to the `main` branch.
    -   You can manually trigger a deployment using `mkdocs gh-deploy`.

4.  **PyPI Publication:**
    -   Building and uploading to PyPI (if applicable):
      ```bash
      python -m build
      python -m twine upload dist/*
      ```

## Automated Documentation

The documentation website ([https://elkins.github.io/synth-pdb](https://elkins.github.io/synth-pdb)) is built using [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

### Key Plugins Used:
-   **mkdocstrings:** Automatically generates API documentation from Python docstrings.
-   **mkdocs-jupyter:** Renders Jupyter notebooks within the documentation.
-   **search:** Provides site-wide search functionality.

To preview the documentation locally:

```bash
mkdocs serve
```

This will start a local server at [http://127.0.0.1:8000](http://127.0.0.1:8000) that automatically reloads when you make changes.
