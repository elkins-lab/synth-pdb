"""
Enforce the philosophy that the code is the textbook via comment density.
Also see test_docs_integrity.py for tests of the educational notes.
"""

import io
import os
import tokenize

import pytest


def calculate_comment_ratio(file_path):
    """
    Calculates the ratio of comment lines (including docstrings) to code lines.
    """
    if not os.path.exists(file_path):
        return 0.0

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    try:
        f_obj = io.StringIO(content)
        tokens = list(tokenize.generate_tokens(f_obj.readline))

        comment_line_indices = set()
        code_line_indices = set()

        for tok in tokens:
            start_line = tok.start[0]
            tok.end[0]

            if tok.type == tokenize.COMMENT:
                # Check if the comment (excluding the #) has non-blank content
                comment_content = tok.string.lstrip("#").strip()
                if comment_content:
                    comment_line_indices.add(start_line)
            elif tok.type == tokenize.STRING:
                # Docstrings/Strings: count only lines with non-blank text
                s_lines = tok.string.splitlines()
                for i, s_line in enumerate(s_lines):
                    if s_line.strip():
                        comment_line_indices.add(start_line + i)
            elif tok.type not in (
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
                tokenize.STRING,
            ):
                # Count actual code tokens (excluding blank lines and docstrings)
                code_line_indices.add(start_line)

        # A line can be both a code line AND a comment line (inline comments)
        final_comment_count = len(comment_line_indices)
        final_code_count = len(code_line_indices)

        return final_comment_count / max(1, final_code_count)
    except Exception:
        return 0.0


@pytest.mark.parametrize(
    "file_path, min_ratio",
    [
        ("synth_pdb/__init__.py", 0.33),
        ("synth_pdb/batch_generator.py", 0.59),
        ("synth_pdb/biophysics.py", 1.14),
        ("synth_pdb/chemical_shifts.py", 0.94),
        ("synth_pdb/cofactors.py", 0.89),
        ("synth_pdb/contact.py", 1.41),
        ("synth_pdb/coupling.py", 1.37),
        ("synth_pdb/data.py", 0.98),
        ("synth_pdb/dataset.py", 0.75),
        ("synth_pdb/decoys.py", 0.72),
        ("synth_pdb/distogram.py", 1.76),
        ("synth_pdb/docking.py", 0.72),
        ("synth_pdb/evolution.py", 1.23),
        ("synth_pdb/export.py", 0.86),
        ("synth_pdb/generator.py", 0.82),
        ("synth_pdb/geometry.py", 1.11),
        ("synth_pdb/j_coupling.py", 5.00),
        ("synth_pdb/main.py", 0.43),
        ("synth_pdb/msa.py", 0.91),
        ("synth_pdb/nef_io.py", 1.08),
        ("synth_pdb/nmr.py", 5.00),
        ("synth_pdb/orientogram.py", 1.13),
        ("synth_pdb/packing.py", 1.04),
        ("synth_pdb/pdb_utils.py", 1.02),
        ("synth_pdb/physics.py", 0.60),
        ("synth_pdb/plm.py", 2.05),
        ("synth_pdb/quality/__init__.py", 0.00),
        ("synth_pdb/quality/classifier.py", 0.68),
        ("synth_pdb/quality/features.py", 0.48),
        ("synth_pdb/quality/gnn/__init__.py", 1.75),
        ("synth_pdb/quality/gnn/gnn_classifier.py", 1.90),
        ("synth_pdb/quality/gnn/graph.py", 1.69),
        ("synth_pdb/quality/gnn/model.py", 2.70),
        ("synth_pdb/quality/interpolate.py", 0.42),
        ("synth_pdb/quality/models/__init__.py", 0.00),
        ("synth_pdb/rdc.py", 18.50),
        ("synth_pdb/relaxation.py", 1.18),
        ("synth_pdb/scoring.py", 1.16),
        ("synth_pdb/special_chemistry.py", 1.85),
        ("synth_pdb/structure_utils.py", 5.00),
        ("synth_pdb/torsion.py", 0.91),
        ("synth_pdb/validator.py", 0.56),
        ("synth_pdb/viewer.py", 0.71),
        ("synth_pdb/visualization.py", 1.05),
    ],
)
def test_library_documentation_density(file_path, min_ratio):
    """
    Enforces a minimum documentation density for core library components.
    This ensures the project maintains its pedagogical and educational value.
    """
    # Adjust path relative to project root if needed
    full_path = os.path.join(os.path.dirname(__file__), "..", file_path)
    if not os.path.exists(full_path):
        # Fallback for different test execution working directories
        full_path = file_path

    ratio = calculate_comment_ratio(full_path)
    assert (
        ratio >= min_ratio
    ), f"Documentation density for {file_path} is {ratio:.2f}, which is below the required {min_ratio}"
