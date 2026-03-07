#!/usr/bin/env python3
"""
Combined fix:
1. Add 4 missing EDUCATIONAL NOTE blocks to physics.py
2. Fix _finalize_output to return True/False so _run_simulation can detect failure
"""
import ast
import io
import sys
import tokenize
from pathlib import Path

PHYSICS = Path(__file__).parent.parent / 'synth_pdb' / 'physics.py'
text = PHYSICS.read_text()

REPLACEMENTS = [
    # ── 1. Topological Validation (inside _preprocess_pdb_for_simulation) ──
    (
        "        topology.createStandardBonds()",
        """\
        # EDUCATIONAL NOTE - Topological Validation:
        # ------------------------------------------
        # OpenMM's PDB reader can sometimes skip bonds if they aren't explicitly
        # in CONECT records or deviate too far from their ideal length.
        # We force bond generation to ensure standard residues have
        # all internal bonds defined, which is required for template matching.
        topology.createStandardBonds()""",
    ),
    # ── 2. Rename "Serialization & Macrocycle Cleanup" → "Serialization:" ──
    (
        "            # EDUCATIONAL NOTE - Serialization & Macrocycle Cleanup:",
        "            # EDUCATIONAL NOTE - Serialization:",
    ),
    # ── 3. Thermal Jiggling note (in _run_simulation orchestrator) ──
    (
        "                if cyclic:\n                    logger.info(\n                        \"Thermal Jiggling: Applying random perturbation to break deadlocks.\"\n                    )",
        """\
                if cyclic:
                    # EDUCATIONAL NOTE - Thermal Jiggling (Simulated Annealing):
                    # ---------------------------------------------------------
                    # Sometimes a linear sequence gets "deadlocked" in a
                    # high-energy conformation that prevents closure.
                    # We apply a brief burst of random motion (perturbation)
                    # followed by another minimization to "jiggle" the
                    # molecule into a closable state.
                    logger.info(
                        "Thermal Jiggling: Applying random perturbation to break deadlocks."
                    )""",
    ),
    # ── 4. Thermal Equilibration note (in _run_simulation orchestrator) ──
    (
        "            # Equilibration steps\n            if equilibration_steps > 0:",
        """\
            # EDUCATIONAL NOTE - Thermal Equilibration (MD):
            # ----------------------------------------------
            # Minimization only finds a "Static Minimum" (0 Kelvin).
            # Real proteins are dynamic. Running MD steps (Langevin Dynamics)
            # resolves clashes and satisfies entropy-driven structural preferences.
            # Equilibration steps
            if equilibration_steps > 0:""",
    ),
    # ── 5. Fix _finalize_output return values ──
    # Change the early-return on empty positions to return False
    (
        "            if len(final_positions) == 0:\n                logger.error(\"OpenMM returned empty positions! Topology might be corrupted.\")\n                return",
        """\
            if len(final_positions) == 0:
                logger.error("OpenMM returned empty positions! Topology might be corrupted.")
                return False""",
    ),
    # Add return True at end of _finalize_output (before the closing of the with block)
    (
        "            final_lines.append(\"END\")\n            f.write(\"\\n\".join(final_lines) + \"\\n\")",
        """\
            final_lines.append("END")
            f.write("\\n".join(final_lines) + "\\n")
        return True""",
    ),
    # ── 6. Fix _run_simulation to check _finalize_output return value ──
    (
        "            # ── Stage 5: Write output PDB ────────────────────────────────────\n            self._finalize_output(\n                output_path, simulation, cyclic, added_bonds,\n                coordination_restraints, hetatm_lines, original_metadata, atom_list,\n            )\n            return final_energy",
        """\
            # ── Stage 5: Write output PDB ────────────────────────────────────
            write_ok = self._finalize_output(
                output_path, simulation, cyclic, added_bonds,
                coordination_restraints, hetatm_lines, original_metadata, atom_list,
            )
            if write_ok is False:
                return None
            return final_energy""",
    ),
]

applied = 0
for old, new in REPLACEMENTS:
    if old not in text:
        print(f"WARNING: snippet not found:\n  {old[:100]!r}")
        continue
    text = text.replace(old, new, 1)
    applied += 1
    print(f"  Applied: {old[:60]!r}")

print(f"\nApplied {applied}/{len(REPLACEMENTS)} replacements.")

try:
    ast.parse(text)
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    sys.exit(1)

PHYSICS.write_text(text)
print("SUCCESS: Written.")

# Check remaining required notes
required = [
    "EDUCATIONAL NOTE - Topological Validation:",
    "EDUCATIONAL NOTE - Thermal Jiggling (Simulated Annealing):",
    "EDUCATIONAL NOTE - Thermal Equilibration (MD):",
    "EDUCATIONAL NOTE - Serialization:",
]
normalized_content = " ".join(text.split())
for r in required:
    normalized_r = " ".join(r.split())
    status = "PRESENT ✓" if normalized_r in normalized_content else "MISSING ✗"
    print(f"  {status}: {r}")

# Report final ratio
content = open(PHYSICS).read()
tokens = list(tokenize.generate_tokens(io.StringIO(content).readline))
cl, code = set(), set()
for tok in tokens:
    sl = tok.start[0]
    if tok.type == tokenize.COMMENT:
        if tok.string.lstrip('#').strip(): cl.add(sl)
    elif tok.type == tokenize.STRING:
        for i, s in enumerate(tok.string.splitlines()):
            if s.strip(): cl.add(sl + i)
    elif tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER, tokenize.STRING):
        code.add(sl)
r = len(cl) / max(1, len(code))
print(f"\nFinal comment ratio: {r:.3f} ({'PASS ✓' if r >= 0.6 else 'FAIL ✗'})")
