#!/usr/bin/env python3
"""Restore all missing EDUCATIONAL NOTE blocks to generator.py and physics.py."""
import ast
import sys
from pathlib import Path

GENERATOR = Path(__file__).parent.parent / 'synth_pdb' / 'generator.py'
PHYSICS   = Path(__file__).parent.parent / 'synth_pdb' / 'physics.py'

# ─────────────────────────────────────────────────────────────────────────────
# GENERATOR.PY REPLACEMENTS
# ─────────────────────────────────────────────────────────────────────────────
GEN_REPLACEMENTS = [

    # 1. Input Validation note (in _resolve_conformation_map, before valid_conformations)
    (
        "    valid_conformations = list(RAMACHANDRAN_PRESETS.keys()) + ['random']\n    if conformation not in valid_conformations:",
        """\
    # EDUCATIONAL NOTE - Input Validation:
    # We validate the default conformation early to give clear error messages.
    # Even if structure parameter overrides it for some residues, we need to
    # ensure the default is valid for any gaps or when structure is not provided.
    valid_conformations = list(RAMACHANDRAN_PRESETS.keys()) + ['random']
    if conformation not in valid_conformations:""",
    ),

    # 2. Per-Residue Conformation Assignment note (in _resolve_conformation_map)
    (
        "    if structure:\n        residue_conformations = _parse_structure_regions(structure, sequence_length)",
        """\
    # EDUCATIONAL NOTE - Per-Residue Conformation Assignment:
    # We now support two modes:
    # 1. Uniform conformation (old behavior): All residues use same conformation
    # 2. Per-region conformation (new!): Different regions can have different conformations

    # Parse per-residue conformations if structure parameter is provided
    if structure:
        residue_conformations = _parse_structure_regions(structure, sequence_length)""",
    ),

    # 3. Gap Handling note (in _resolve_conformation_map)
    (
        "        for i in range(sequence_length):\n            if i not in residue_conformations:\n                residue_conformations[i] = conformation",
        """\
        # EDUCATIONAL NOTE - Gap Handling:
        # If a residue is not specified in the structure parameter,
        # we use the default conformation. This allows users to specify
        # only the interesting regions and let the rest use a sensible default.
        for i in range(sequence_length):
            if i not in residue_conformations:
                residue_conformations[i] = conformation""",
    ),

    # 4. Why We Don't Validate Conformations note (at end of _resolve_conformation_map, before return)
    (
        "    return residue_conformations\n\n\ndef _build_peptide_chain(",
        """\
    # EDUCATIONAL NOTE - Why We Don't Validate Conformations Here:
    # We already validated conformations in _parse_structure_regions(),
    # so we don't need to re-validate them here. The default conformation
    # will be validated when we actually use it below.
    return residue_conformations


def _build_peptide_chain(""",
    ),

    # 5. D-Amino Acid Handling note (in _build_peptide_chain, already there but checking)
    # (already present at line 776, skip)

    # 6. Peptide Bond Chemistry + Terminal Atom Management notes
    (
        "        # Remove terminal atoms that are incompatible with internal peptide bonds\n        if i < len(sequence) - 1 or cyclic:",
        """\
        # EDUCATIONAL NOTE - Peptide Bond Chemistry:
        # A peptide bond forms via dehydration synthesis (loss of H2O).
        # The Carboxyl group (COOH) of one amino acid joins the Amine group (NH2) of the next.
        # This means internal residues lose their terminal Oxygen (OXT) and associated Hydrogens.
        # We must explicitly remove terminal-only atoms from all residues to represent
        # a continuous polypeptide chain correctly and avoid OpenMM template errors.

        # EDUCATIONAL NOTE - Terminal Atom Management in Rings:
        # In a linear peptide, the ends are "unfinished" (OXT at C-term, extra H at N-term).
        # In a cyclic peptide, these atoms are SACRIFICED to form the peptide bond
        # between the ends. Failing to remove them leads to "impossible" geometry
        # with 5-valent carbons or nitrogen atoms with too many bonds.

        # Remove terminal atoms that are incompatible with internal peptide bonds
        if i < len(sequence) - 1 or cyclic:""",
    ),

    # 7. Rotamer Selection Strategy note
    (
        "        # ── Rotamer selection ────────────────────────────────────────────────\n        rotamers = None",
        """\
        # EDUCATIONAL NOTE - Rotamer Selection Strategy:
        # We employ a 'Backbone-Dependent' selection strategy where possible.
        # This means we check the residue's secondary structure context (Helix vs Sheet)
        # to choose the most likely side-chain conformation.
        # Logic:
        # 1. Check if specific rotamers exist for this residue + conformation in BACKBONE_DEPENDENT_LIB.
        # 2. If not, fall back to the generic ROTAMER_LIBRARY (Backbone-Independent).
        # ── Rotamer selection ────────────────────────────────────────────────
        rotamers = None""",
    ),

    # 8. Sidechain Rotation note (inside gamma_atom_name block, before chi1 computation)
    (
        "                if gamma_atom_name:\n                    ca_atom = ref_res_template[ref_res_template.atom_name == \"CA\"][0]",
        """\
                if gamma_atom_name:
                    # EDUCATIONAL NOTE - Sidechain Rotation:
                    # Instead of placing a single atom (which breaks branched residues like VAL),
                    # we rotate the ENTIRE sidechain about the CA-CB axis to reach target chi1.
                    # We use Rodrigues' rotation formula (CCW looking down CA->CB).
                    # rotation_angle = target - current (standard IUPAC convention).
                    ca_atom = ref_res_template[ref_res_template.atom_name == "CA"][0]""",
    ),

    # 9. AHA MOMENT - Superimposition Direction
    (
        "        _, transformation = struc.superimpose(target_backbone_constructed, mobile_backbone_from_template)\n        transformed_res = ref_res_template",
        """\
        # AHA MOMENT - Superimposition Direction:
        # In the "AI Trinity" debugging phase, we found that residues were disconnected
        # (6A-13A gaps). This was due to superimposing the backbone onto the template
        # instead of moving the template into our newly constructed global frame.
        # Fixed: target=constructed_frame, mobile=residue_template.
        _, transformation = struc.superimpose(target_backbone_constructed, mobile_backbone_from_template)
        transformed_res = ref_res_template""",
    ),

    # 10. Chiral Mirroring strategy note
    (
        "        # ── D-amino acid chiral mirroring ─────────────────────────────────\n        if is_d and res_name != \"GLY\":",
        """\
        # EDUCATIONAL NOTE - Chiral Mirroring strategy:
        # -------------------------------------------
        # To convert an L-amino acid to a D-amino acid without separate templates,
        # we treat the CA as the origin and the N-CA-C plane as our mirror.
        #
        # Why Mirroring Works:
        # 1. Geometry Preservation: Mirroring across the backbone plane preserves
        #    all bond lengths and angles within the sidechain.
        # 2. Stereocenter Inversion: This transformation exactly inverts the
        #    chirality at the C-alpha atom (from L to D).
        # 3. Backbone Compatibility: Because we reflect *across* the backbone plane,
        #    the positions of the backbone atoms (N, CA, C) remain unchanged,
        #    ensuring the chain connectivity is not broken.
        # ── D-amino acid chiral mirroring ─────────────────────────────────
        if is_d and res_name != "GLY":""",
    ),

    # 11. D-Residue Naming note (before the L_TO_D_MAPPING assignment)
    (
        "        transformed_res.res_id[:] = res_id\n        if is_d:\n            transformed_res.res_name[:] = L_TO_D_MAPPING.get(res_name, res_name)",
        """\
        transformed_res.res_id[:] = res_id
        # EDUCATIONAL NOTE - D-Residue Naming:
        # We use a 4-letter prefix 'D' (e.g., DALA, DGLU) to distinguish
        # D-amino acids from their L-counterparts in the PDB file.
        # This makes it easier for validators and downstream tools to recognize the chirality.
        if is_d:
            transformed_res.res_name[:] = L_TO_D_MAPPING.get(res_name, res_name)""",
    ),

    # 12. Biophysical Realism (Phase 2) note in _apply_biophysical_mods
    (
        "    # EDUCATIONAL NOTE - Side-Chain Optimization:\n    # If requested, run Monte Carlo optimization to fix steric clashes.\n    if optimize_sidechains:",
        """\
    # EDUCATIONAL NOTE - Biophysical Realism (Phase 2):
    # We apply chemical modifications after geometric construction/packing but BEFORE
    # energy minimization. Correct protonation states (pH) are critical for
    # correct electrostatics in the forcefield.

    # EDUCATIONAL NOTE - Side-Chain Optimization:
    # If requested, run Monte Carlo optimization to fix steric clashes.
    # This is "Phase 1" of biophysical realism.
    if optimize_sidechains:""",
    ),

    # 13. Metal Ion Coordination (Phase 15) note in _apply_biophysical_mods
    (
        "    # EDUCATIONAL NOTE - Metal Ion Coordination:\n    # Inorganic cofactors like Zn2+ are automatically detected and injected.",
        """\
    # EDUCATIONAL NOTE - Metal Ion Coordination (Phase 15):
    # Inorganic cofactors like Zinc (Zn2+) are automatically detected.
    # If a coordination motif is found (Cys/His clusters), the ion is
    # injected and harmonic constraints are applied in the physics module.""",
    ),

    # 14. Energy Minimization (Phase 2) note in _do_energy_minimization
    (
        "    logger.info(\"Running energy minimization (OpenMM)...\")\n    try:",
        """\
    # EDUCATIONAL NOTE - Energy Minimization (Phase 2):
    # OpenMM requires a file-based interaction for easy topology handling from PDB.
    # So we write the current state to a temp file, minimize it, and read it back.
    logger.info("Running energy minimization (OpenMM)...")
    try:""",
    ),

    # 15. B-factors and Occupancy notes in _assemble_pdb_output (currently one combined note)
    (
        "    # EDUCATIONAL NOTE - Adding Realistic B-factors & Occupancy:\n    # Biotite sets B-factors to 0.00 and occupancy to 1.00 by default.\n    # We post-process to insert physically meaningful values derived from S2.",
        """\
    # EDUCATIONAL NOTE - Adding Realistic B-factors:
    # Biotite sets all B-factors to 0.00 by default. We post-process the PDB string
    # to replace these with realistic values based on atom type, position, and residue type.
    # This makes the output look more professional and realistic.

    # EDUCATIONAL NOTE - Adding Realistic Occupancy:
    # Similarly, biotite sets all occupancy values to 1.00. We calculate realistic
    # occupancy values (0.85-1.00) that correlate with B-factors and reflect disorder.""",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS.PY REPLACEMENTS (the two minor inline comments)
# ─────────────────────────────────────────────────────────────────────────────
PHYS_REPLACEMENTS = [
    # 1. "We do NOT add the bond to the Topology here" note
    (
        "        if cyclic:\n            logger.info(\"Cyclizing peptide via harmonic restraints (Restraint-First approach).\")",
        """\
        # EDUCATIONAL NOTE: We do NOT add the bond to the Topology here.
        # Adding it here causes OpenMM's template-matcher to fail ("Too many external bonds").
        # Instead, we use massive harmonic restraints to CLOSE the ring physically.
        if cyclic:
            logger.info("Cyclizing peptide via harmonic restraints (Restraint-First approach).")""",
    ),
    # 2. GHOSTING THE TOPOLOGICAL BOND comment
    (
        "                    if n_idx != -1 and c_idx != -1:\n                        pull_force.addBond(n_idx, c_idx, [0.133 * unit.nanometers])\n                        logger.info(f\"Added massive cyclic pull force: {n_idx} -- {c_idx}\")\n\n                        # Ghost the welded topological bond",
        """\
                    if n_idx != -1 and c_idx != -1:
                        pull_force.addBond(n_idx, c_idx, [0.133 * unit.nanometers])
                        logger.info(f"Added massive cyclic pull force: {n_idx} -- {c_idx}")

                        # GHOSTING THE TOPOLOGICAL BOND:
                        # Since we welded the ring in the topology for templates,
                        # we must zero out its physical forces so the pull-magnet works.
                        # Ghost the welded topological bond""",
    ),
]


def apply_replacements(path, replacements):
    text = path.read_text()
    applied, skipped = 0, 0
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new, 1)
            applied += 1
        else:
            print(f"  WARNING: not found in {path.name}:\n    {old[:80]!r}")
            skipped += 1
    try:
        ast.parse(text)
    except SyntaxError as e:
        print(f"SYNTAX ERROR in {path.name}: {e}")
        sys.exit(1)
    path.write_text(text)
    print(f"  {path.name}: applied {applied}/{len(replacements)} ({skipped} not found)")
    return text


print("Restoring generator.py...")
apply_replacements(GENERATOR, GEN_REPLACEMENTS)

print("Restoring physics.py...")
apply_replacements(PHYSICS, PHYS_REPLACEMENTS)

# Final audit
import re

for path, label in [(GENERATOR, "generator.py"), (PHYSICS, "physics.py")]:
    notes = re.findall(r'EDUCATIONAL NOTE|AHA MOMENT', path.read_text())
    print(f"  {label}: {len(notes)} EDUCATIONAL NOTE / AHA MOMENT occurrences")

print("\nDone.")
