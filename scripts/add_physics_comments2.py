#!/usr/bin/env python3
"""Add the remaining ~40 comment lines to reach 0.60 ratio in physics.py."""
import ast
import io
import sys
import tokenize
from pathlib import Path

PHYSICS = Path(__file__).parent.parent / "synth_pdb" / "physics.py"
text = PHYSICS.read_text()

REPLACEMENTS = [
    # _preprocess_pdb_for_simulation — HETATM stripping comment
    (
        "                    if res_name_upper in ['ZN', 'FE', 'MG', 'CA', 'NA', 'CL']:",
        """\
                    # EDUCATIONAL NOTE - Ion Stripping:
                    # Ions like Zn2+, Fe2+, Mg2+ crash Modeller.addHydrogens()
                    # because they have no hydrogen template. We stash them in
                    # hetatm_lines and re-append them after minimization.
                    if res_name_upper in ['ZN', 'FE', 'MG', 'CA', 'NA', 'CL']:""",
    ),
    # _setup_openmm_modeller — addHydrogens comment
    (
        "        # Add hydrogens\n        if add_hydrogens:",
        """\
        # EDUCATIONAL NOTE - Why Add Hydrogens?
        # X-ray crystallography often doesn't resolve hydrogen atoms because they
        # have very few electrons. However, Molecular Dynamics forcefields (like
        # Amber) are explicitly "All-Atom". They REQUIRE hydrogens to calculate
        # bond angles and electrostatics (h-bonds) correctly.
        #
        # NMR Perspective: Unlike X-ray, NMR relies entirely on the magnetic spin
        # of protons (H1). Correctly placing them is critical not just for physics
        # but for predicting NOEs (Nuclear Overhauser Effects) which depend on
        # H-H distances. We use `app.Modeller` to "guess" the standard positions
        # of hydrogens at specific pH (7.0).
        # Add hydrogens
        if add_hydrogens:""",
    ),
    # _build_simulation_context — system creation comment
    (
        "        # System creation",
        """\
        # EDUCATIONAL NOTE - Anatomy of a Forcefield:
        # A forcefield (like Amber14) approximates the potential energy (U) of a
        # molecule as a sum of four main terms:
        #   U = U_bond + U_angle + U_torsion + [U_vdw + U_elec]
        # Minimization finds the coordinate set where dU/dX = 0.
        # System creation""",
    ),
    # _finalize_output — CONECT comment
    (
        "            # Force CONECT for disulfides",
        """\
            # EDUCATIONAL NOTE - CONECT Records & Visualization:
            # CONECT records are critical for molecular viewers (PyMOL, Chimera)
            # to draw covalent bonds that OpenMM's PDB writer may not emit
            # automatically for non-standard connections (SS bonds, metal–ligand).
            # We enumerate them from the final topology and write both directions.
            # Force CONECT for disulfides""",
    ),
]

applied = 0
for old, new in REPLACEMENTS:
    if old not in text:
        print(f"WARNING: snippet not found:\n  {old[:80]!r}")
        continue
    text = text.replace(old, new, 1)
    applied += 1

print(f"Applied {applied}/{len(REPLACEMENTS)} replacements.")

try:
    ast.parse(text)
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    sys.exit(1)

PHYSICS.write_text(text)
print("SUCCESS: Written.")


# Report new ratio
def ratio(path):
    content = open(path).read()
    tokens = list(tokenize.generate_tokens(io.StringIO(content).readline))
    cl, code = set(), set()
    for tok in tokens:
        sl = tok.start[0]
        if tok.type == tokenize.COMMENT:
            if tok.string.lstrip("#").strip():
                cl.add(sl)
        elif tok.type == tokenize.STRING:
            for i, s in enumerate(tok.string.splitlines()):
                if s.strip():
                    cl.add(sl + i)
        elif tok.type not in (
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENDMARKER,
            tokenize.STRING,
        ):
            code.add(sl)
    return len(cl), len(code), len(cl) / max(1, len(code))


c, cd, r = ratio(PHYSICS)
print(f"  New ratio: {r:.3f}  (comments={c}, code={cd})")
print(f"  {'PASS ✓' if r >= 0.6 else f'FAIL — need {int(0.6*cd - c)} more'}")
