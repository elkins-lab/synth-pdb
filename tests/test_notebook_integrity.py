"""Static integrity checks for tutorial notebooks."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRS = [
    ROOT / "docs" / "tutorials",
    ROOT / "examples" / "interactive_tutorials",
    ROOT / "examples" / "ml_integration",
    ROOT / "examples" / "ml_loading",
]


def _notebooks() -> list[Path]:
    notebooks: list[Path] = []
    for directory in NOTEBOOK_DIRS:
        if directory.exists():
            notebooks.extend(sorted(directory.glob("*.ipynb")))
    return notebooks


def test_tutorial_notebooks_are_valid_json() -> None:
    """Notebook JSON corruption should be caught before publication."""
    assert _notebooks(), "No tutorial notebooks found"

    for notebook in _notebooks():
        with notebook.open(encoding="utf-8") as handle:
            json.load(handle)


def test_docs_tutorial_markdown_images_exist() -> None:
    """Local images referenced by docs tutorials must ship with the docs."""
    image_pattern = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
    docs_dir = ROOT / "docs" / "tutorials"

    for notebook in sorted(docs_dir.glob("*.ipynb")):
        data = json.loads(notebook.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", ""))
            for cell in data.get("cells", [])
            if cell.get("cell_type") == "markdown"
        )
        for image_ref in image_pattern.findall(source):
            if image_ref.startswith(("http://", "https://")):
                continue
            assert (notebook.parent / image_ref).exists(), (
                f"{notebook.relative_to(ROOT)} references missing image {image_ref}"
            )
