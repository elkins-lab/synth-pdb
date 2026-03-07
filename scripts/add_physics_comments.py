#!/usr/bin/env python3
"""Add educational comment blocks back to the 4 new physics.py helper methods."""
import ast
import sys
from pathlib import Path

PHYSICS = Path(__file__).parent.parent / 'synth_pdb' / 'physics.py'
text = PHYSICS.read_text()

# ── Each replacement is (old_snippet, new_snippet) ───────────────────────────
REPLACEMENTS = [
    # ── _preprocess_pdb_for_simulation ────────────────────────────────────────
    (
        "        ptm_map = {",
        """\
        # EDUCATIONAL NOTE - PDB PRE-PROCESSING (OpenMM Template Fix):
        # -----------------------------------------------------------
        # OpenMM's standard forcefields (amber14-all) are highly optimized for wild-type
        # human proteins but frequently lack templates for:
        # 1. Phosphorylated residues (SEP, TPO, PTR)
        # 2. Histidine tautomers (HIE, HID) named explicitly in the input.
        # 3. D-Amino Acids (DAL, DPH, etc.) - These require L-analog templates.
        #
        # To prevent "No template found" errors, we surgically rename residues to
        # their standard counterparts BEFORE loading. We preserve the original
        # identity in `original_metadata` for final restoration.
        ptm_map = {""",
    ),
    (
        "            n_term_serial, c_term_serial = None, None",
        """\
            # EDUCATIONAL NOTE - Cyclic CONECT Stripping:
            # OpenMM's PDB reader creates CONECT records for all explicit bonds.
            # For cyclic peptides, the head-to-tail bond is already encoded as
            # a CONECT (written by generator.py). We must remove it here so
            # addHydrogens later does not see a conflicting terminal N–C bond.
            n_term_serial, c_term_serial = None, None""",
    ),
    (
        "            # Add dummy OXT for cyclic peptides to satisfy C-terminal templates",
        """\
            # EDUCATIONAL NOTE - Dummy OXT Insertion:
            # OpenMM amber14 residue templates for C-termini require an OXT
            # oxygen to match the "C_TERM" patch. Cyclic peptides lack this atom
            # so we add a temporary OXT positioned ~1.2 Å from the terminal C.
            # physics.py _finalize_output() will delete it after minimization.
            # Add dummy OXT for cyclic peptides to satisfy C-terminal templates""",
    ),
    # ── _setup_openmm_modeller ────────────────────────────────────────────────
    (
        "        # Heuristic backbone stitching",
        """\
        # EDUCATIONAL NOTE - Topology Bridging (welding the ring):
        # --------------------------------------------------------
        # Adding the head-to-tail bond to the Topology triggers OpenMM's Amber
        # template matcher to look for non-existent cyclic templates or
        # specialized patches, leading to "Too many external bonds" errors.
        # We use Restraints instead (applied later in _build_simulation_context).
        #
        # EDUCATIONAL NOTE - Robust Backbone Stitching (Heuristic Bonding):
        # -----------------------------------------------------------------
        # When building de novo structures, standard PDB-to-Topology builders
        # often miss local connectivity. We implement a "Heuristic Welder" that
        # looks for missing C-N peptide bonds based on proximity. If two
        # sequential residues are close but unbonded, we manually weld their
        # backbone atoms to ensure a continuous, force-propagating chain.
        # Heuristic backbone stitching""",
    ),
    (
        "        # Disulfide bond detection by SG proximity",
        """\
        # EDUCATIONAL NOTE - The SSBOND Capture Radius:
        # ---------------------------------------------
        # Unlike distance-based bonding in simple geometry, physical disulfide
        # formation is highly sensitive to the S-S distance (~2.03 Å).
        # We use a large "Capture Radius" (SSBOND_CAPTURE_RADIUS) to detect
        # potential pairs in un-optimized structures, then allow the "Mega-Pull"
        # to bring them into the ideal covalent distance.
        # Disulfide bond detection by SG proximity""",
    ),
    (
        "        # Salt bridge + metal coordination detection via biotite",
        """\
        # EDUCATIONAL NOTE - Salt Bridges & Electrostatics:
        # -------------------------------------------------
        # A Salt Bridge is an electrostatic attraction between a cationic sidechain
        # (e.g. Lysine/Arginine) and an anionic one (Aspartate/Glutamate).
        # Forcefields model these naturally via Coulomb's law, but in vacuum
        # simulations, the attraction can be artificially weak or slow to form.
        # We apply harmonic "Bungee" restraints to help these bridges snap together.
        # Salt bridge + metal coordination detection via biotite""",
    ),
    (
        "        # CYX renaming + HG deletion for bonded cysteines",
        """\
        # EDUCATIONAL NOTE - CYX Renaming & Thiol Stripping:
        # -------------------------------------------------
        # In classical forcefields, a standard Cysteine (CYS) has a thiol group (-SH).
        # When a disulfide bond (S-S) forms, two hydrogens are LOST.
        # OpenMM's Amber forcefield uses a separate residue template ('CYX') for
        # these bonded cysteines. We must rename the residues AND manually delete
        # the HG atoms, or the physics engine will see a "template mismatch" error.
        # CYX renaming + HG deletion for bonded cysteines""",
    ),
    # ── _build_simulation_context ─────────────────────────────────────────────
    (
        "        # Cyclic terminal ghosting + shadow cap zeroing",
        """\
        # EDUCATIONAL NOTE - The "Nuclear Option" & "Shadow Caps":
        # -------------------------------------------------------
        # Closing a ring is a physical paradox for most forcefields. The N and C
        # termini are parameterized as charged ions that repel each other violently.
        #
        # 1. Terminal Ghosting: We surgically disable all non-bonded interactions
        #    between the first and last residues. They can now pass through each other
        #    without steric or electrostatic resistance.
        #
        # 2. Shadow Caps: To satisfy OpenMM's template requirements, we temporarily
        #    attached ACE/NME dummy residues. Here, we zero out ALL their forces.
        #    They allow the simulation to run but contribute nothing to the energy,
        #    leaving the path clear for the "Mega-Pull" to snap the ring shut.
        # EDUCATIONAL NOTE - Constraints and Macrocycles:
        # For macrocycles, we temporarily DISABLE all constraints (like HBonds)
        # to allow the chain to bend freely into a ring during the pull phase.
        # We also use vacuum (NoCutoff) to maximize closure speed.
        # Cyclic terminal ghosting + shadow cap zeroing""",
    ),
    (
        "        # Pull forces for cyclic closure and disulfide formation",
        """\
        # EDUCATIONAL NOTE - Harmonic "Pull" Restraints & Hard Constraints:
        # -----------------------------------------------------------------
        # To bridge the gap between N and C termini, we use two levels of force:
        # 1. Harmonic Pull: A massive "spring" (100M kJ/mol/nm²) that treats the
        #    termini like two magnets. It provides a global gradient that pulls
        #    the structure toward closure.
        # 2. Hard Constraint: A specialized OpenMM constraint that FIXES the
        #    distance at exactly 1.33 Angstroms. While the pull force gets us close,
        #    the constraint ensures the final "weld" satisfies the perfect geometry
        #    required by downstream NMR tools.
        #
        # EDUCATIONAL NOTE - Why we avoid adding a hard constraint initially:
        # If the termini are far apart, a hard constraint crashes the system.
        # The 100M kJ magnet (pull_force) will get us to 1.33Å first.
        # Pull forces for cyclic closure and disulfide formation""",
    ),
    # ── _finalize_output ──────────────────────────────────────────────────────
    (
        "            # Restore original residue names and IDs",
        """\
            # EDUCATIONAL NOTE - Serialization & Macrocycle Cleanup:
            # -------------------------------------------------------
            # After physics completes, we must "tidy up" our synthetic hack.
            # We prune the "Shadow Caps" (ACE/NME) and any extra terminal hydration
            # protons (H1, H2, H3, OXT) that Modeller added. We rename the remaining
            # amide proton to 'H' to satisfy canonical PDB naming. Finally, we
            # project the original residue names and IDs back onto the physics-optimized
            # coordinates, bridging the gap between molecular physics and structural
            # metadata (PTMs, D-amino acids).
            # Restore original residue names and IDs""",
    ),
]

for old, new in REPLACEMENTS:
    if old not in text:
        print(f"WARNING: Could not find snippet:\n  {old[:80]!r}")
        continue
    text = text.replace(old, new, 1)

try:
    ast.parse(text)
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    sys.exit(1)

PHYSICS.write_text(text)
print("SUCCESS: Educational comments added to physics.py.")

# Report new ratio
import io
import tokenize


def ratio(path):
    content = open(path).read()
    tokens = list(tokenize.generate_tokens(io.StringIO(content).readline))
    cl, code = set(), set()
    for tok in tokens:
        sl = tok.start[0]
        if tok.type == tokenize.COMMENT:
            if tok.string.lstrip('#').strip(): cl.add(sl)
        elif tok.type == tokenize.STRING:
            for i, s in enumerate(tok.string.splitlines()):
                if s.strip(): cl.add(sl+i)
        elif tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER, tokenize.STRING):
            code.add(sl)
    return len(cl), len(code), len(cl)/max(1,len(code))
c, cd, r = ratio(PHYSICS)
print(f"  New ratio: {r:.3f}  (comments={c}, code={cd})")
print(f"  {'PASS' if r >= 0.6 else 'FAIL'}: target 0.60")
