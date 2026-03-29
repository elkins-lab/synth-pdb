#!/usr/bin/env python3
"""Add missing type annotations to all synth_pdb source files.
Uses exact string replacement on function signatures — no logic changes.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent / "synth_pdb"

PATCHES: dict[str, list[tuple[str, str]]] = {}

# ─────────────────────────────────────────────────────────────────────────────
# physics.py
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["physics.py"] = [
    # __init__
    (
        "def __init__(self, forcefield_name='amber14-all.xml', solvent_model='app.OBC2', box_size=1.0):",
        "def __init__(self, forcefield_name: str = 'amber14-all.xml', solvent_model: str = 'app.OBC2', box_size: float = 1.0) -> None:",
    ),
    # minimize
    (
        "def minimize(self, pdb_file_path, output_path, max_iterations=0, tolerance=10.0, cyclic=False, disulfides=None, coordination=None):",
        "def minimize(self, pdb_file_path: str, output_path: str, max_iterations: int = 0, tolerance: float = 10.0, cyclic: bool = False, disulfides: Optional[List] = None, coordination: Optional[List] = None) -> bool:",
    ),
    # equilibrate
    (
        "def equilibrate(self, pdb_file_path, output_path, steps=1000, cyclic=False, disulfides=None, coordination=None):",
        "def equilibrate(self, pdb_file_path: str, output_path: str, steps: int = 1000, cyclic: bool = False, disulfides: Optional[List] = None, coordination: Optional[List] = None) -> bool:",
    ),
    # add_hydrogens_and_minimize
    (
        "def add_hydrogens_and_minimize(self, pdb_file_path, output_path, max_iterations=0, tolerance=10.0, cyclic=False, disulfides=None, coordination=None):",
        "def add_hydrogens_and_minimize(self, pdb_file_path: str, output_path: str, max_iterations: int = 0, tolerance: float = 10.0, cyclic: bool = False, disulfides: Optional[List] = None, coordination: Optional[List] = None) -> bool:",
    ),
    # calculate_energy — input_data unannotated
    (
        "def calculate_energy(self, input_data, cyclic=False) -> float:",
        "def calculate_energy(self, input_data: Union[str, Any], cyclic: bool = False) -> Optional[float]:",
    ),
    # _create_system_robust
    (
        "def _create_system_robust(self, topology, constraints, modeller=None):",
        "def _create_system_robust(self, topology: Any, constraints: Any, modeller: Optional[Any] = None) -> Tuple[Any, Any, Any]:",
    ),
    # _preprocess_pdb_for_simulation
    (
        "def _preprocess_pdb_for_simulation(self, input_path, cyclic, disulfides_param):",
        "def _preprocess_pdb_for_simulation(self, input_path: str, cyclic: bool, disulfides_param: Optional[List]) -> Tuple[Any, Any, List[str], Dict[Any, Any]]:",
    ),
    # _setup_openmm_modeller (multi-line signature)
    (
        "    def _setup_openmm_modeller(\n        self, topology, positions, add_hydrogens, cyclic, coordination_param, atom_list\n    ):",
        "    def _setup_openmm_modeller(\n        self, topology: Any, positions: Any, add_hydrogens: bool,\n        cyclic: bool, coordination_param: Optional[List], atom_list: List[Any]\n    ) -> Tuple[Any, List, List, List, List[Any]]:",
    ),
    # _build_simulation_context (multi-line signature)
    (
        "    def _build_simulation_context(\n        self, modeller, cyclic, added_bonds, salt_bridge_restraints,\n        coordination_restraints, atom_list\n    ):",
        "    def _build_simulation_context(\n        self, modeller: Any, cyclic: bool, added_bonds: List,\n        salt_bridge_restraints: List, coordination_restraints: List, atom_list: List[Any]\n    ) -> Tuple[Any, Any, int, int, Any, Any]:",
    ),
    # _finalize_output (multi-line signature)
    (
        "    def _finalize_output(\n        self, output_path, simulation, cyclic, added_bonds,\n        coordination_restraints, hetatm_lines, original_metadata, atom_list\n    ):",
        "    def _finalize_output(\n        self, output_path: str, simulation: Any, cyclic: bool, added_bonds: List,\n        coordination_restraints: List, hetatm_lines: List[str],\n        original_metadata: Dict[Any, Any], atom_list: List[Any]\n    ) -> Optional[bool]:",
    ),
    # _run_simulation
    (
        "def _run_simulation(self, input_path, output_path, max_iterations=0, tolerance=10.0, add_hydrogens=True, equilibration_steps=0, cyclic=False, disulfides=None, coordination=None):",
        "def _run_simulation(self, input_path: str, output_path: str, max_iterations: int = 0, tolerance: float = 10.0, add_hydrogens: bool = True, equilibration_steps: int = 0, cyclic: bool = False, disulfides: Optional[List] = None, coordination: Optional[List] = None) -> Optional[float]:",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# validator.py
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["validator.py"] = [
    # __init__ — params already typed, just add return
    (
        "def __init__(self, pdb_content: Optional[str] = None, parsed_atoms: Optional[List[Dict[str, Any]]] = None):",
        "def __init__(self, pdb_content: Optional[str] = None, parsed_atoms: Optional[List[Dict[str, Any]]] = None) -> None:",
    ),
    # validate_bond_lengths
    (
        "def validate_bond_lengths(self, tolerance: float = 0.05):",
        "def validate_bond_lengths(self, tolerance: float = 0.05) -> Dict[str, Any]:",
    ),
    # validate_bond_angles
    (
        "def validate_bond_angles(self, tolerance: float = 5.0):",
        "def validate_bond_angles(self, tolerance: float = 5.0) -> Dict[str, Any]:",
    ),
    # validate_ramachandran
    (
        "def validate_ramachandran(self):",
        "def validate_ramachandran(self) -> Dict[str, Any]:",
    ),
    # validate_steric_clashes (multi-line)
    (
        "    def validate_steric_clashes(\n        self,\n        min_atom_distance: float = 0.5,\n        min_ca_distance: float = 3.0,",
        "    def validate_steric_clashes(\n        self,\n        min_atom_distance: float = 0.5,\n        min_ca_distance: float = 3.0,",
    ),
    # validate_peptide_plane
    (
        "def validate_peptide_plane(self, tolerance_deg: float = 30.0):",
        "def validate_peptide_plane(self, tolerance_deg: float = 30.0) -> Dict[str, Any]:",
    ),
    # validate_side_chain_rotamers
    (
        "def validate_side_chain_rotamers(self, tolerance: float = 40.0):",
        "def validate_side_chain_rotamers(self, tolerance: float = 40.0) -> Dict[str, Any]:",
    ),
    # calculate_dihedrals — input_data unannotated
    (
        "def calculate_dihedrals(self, input_data=None) -> Dict[str, List[float]]:",
        "def calculate_dihedrals(self, input_data: Optional[str] = None) -> Dict[str, List[float]]:",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# generator.py  (PeptideGenerator and PeptideResult classes)
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["generator.py"] = [
    # PeptideGenerator.__init__
    (
        'def __init__(self, sequence: str = "ALA-GLY-SER", **kwargs):',
        'def __init__(self, sequence: str = "ALA-GLY-SER", **kwargs: Any) -> None:',
    ),
    # PeptideResult.__init__
    (
        "def __init__(self, pdb_content: str):",
        "def __init__(self, pdb_content: str) -> None:",
    ),
    # PeptideResult.save
    (
        "def save(self, path: str):",
        "def save(self, path: str) -> None:",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# dataset.py
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["dataset.py"] = [
    (
        "def _generate_single_sample_task(args):",
        "def _generate_single_sample_task(args: tuple) -> Optional[Dict[str, Any]]:",
    ),
    (
        "def _generate_single_sample_npz_task(args):",
        "def _generate_single_sample_npz_task(args: tuple) -> Optional[Dict[str, Any]]:",
    ),
    (
        "def prepare_directories(self):",
        "def prepare_directories(self) -> None:",
    ),
    (
        "def generate(self):",
        "def generate(self) -> None:",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# decoys.py
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["decoys.py"] = [
    (
        "def __init__(self):\n        pass",
        "def __init__(self) -> None:\n        pass",
    ),
    (
        "def _extract_backbone_dihedrals(self, pdb_content: str):",
        "def _extract_backbone_dihedrals(self, pdb_content: str) -> Dict[str, List[float]]:",
    ),
    (
        "def _shuffle_pdb_sequence(self, pdb_content: str):",
        "def _shuffle_pdb_sequence(self, pdb_content: str) -> str:",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# geometry.py
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["geometry.py"] = [
    (
        "def njit(func=None, **kwargs):",
        "def njit(func: Optional[Any] = None, **kwargs: Any) -> Any:",
    ),
    (
        "def normalize(v):",
        "def normalize(v: np.ndarray) -> np.ndarray:",
    ),
    (
        "def rotate_points(points, axis_p1, axis_p2, angle_deg):",
        "def rotate_points(points: np.ndarray, axis_p1: np.ndarray, axis_p2: np.ndarray, angle_deg: float) -> np.ndarray:",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# batch_generator.py
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["batch_generator.py"] = [
    (
        "def save_pdb(self, path: str, index: int = 0):",
        "def save_pdb(self, path: str, index: int = 0) -> None:",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# packing.py
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["packing.py"] = [
    (
        "def __init__(self, steps: int = 500, temperature: float = 0.5):",
        "def __init__(self, steps: int = 500, temperature: float = 0.5) -> None:",
    ),
    (
        "def optimize_sidechains(peptide: struc.AtomArray, steps=500) -> struc.AtomArray:",
        "def optimize_sidechains(peptide: struc.AtomArray, steps: int = 500) -> struc.AtomArray:",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# biophysics.py
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["biophysics.py"] = [
    (
        "def find_salt_bridges(structure: struc.AtomArray, cutoff: float = 5.0)",
        "def find_salt_bridges(structure: struc.AtomArray, cutoff: float = 5.0) -> List[Dict[str, Any]]",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# contact.py
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["contact.py"] = [
    (
        'def compute_contact_map(structure, method="ca", threshold=8.0, power=None):',
        'def compute_contact_map(structure: Any, method: str = "ca", threshold: float = 8.0, power: Optional[float] = None) -> np.ndarray:',
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# torsion.py  (the function already has typed params, just needs return)
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["torsion.py"] = [
    (
        "def export_torsion_angles(data: List[Dict[str, Any]], output_file: str",
        "def export_torsion_angles(data: List[Dict[str, Any]], output_file: str",
        # Handled below via special approach — return type appended
    ),
]
# Reset — will handle in the script logic instead
PATCHES["torsion.py"] = []

# ─────────────────────────────────────────────────────────────────────────────
# distogram.py  (same: params typed, return missing)
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["distogram.py"] = []

# ─────────────────────────────────────────────────────────────────────────────
# evolution.py
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["evolution.py"] = [
    (
        "def write_msa(sequences: List[str], filename: str):",
        "def write_msa(sequences: List[str], filename: str) -> None:",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# docking.py
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["docking.py"] = [
    (
        "def __init__(self, forcefield_name='amber14-all.xml'):",
        "def __init__(self, forcefield_name: str = 'amber14-all.xml') -> None:",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# plm.py
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["plm.py"] = [
    (
        "def _extract_sequence(structure) -> str:",
        "def _extract_sequence(structure: Any) -> str:",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# quality/classifier.py
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["quality/classifier.py"] = [
    (
        "def __init__(self, model_path: Optional[str] = None):",
        "def __init__(self, model_path: Optional[str] = None) -> None:",
    ),
    (
        "def load_model(self, path: str):",
        "def load_model(self, path: str) -> None:",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# quality/interpolate.py
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["quality/interpolate.py"] = [
    (
        "def _reconstruct_backbone(phi, psi, omega):",
        "def _reconstruct_backbone(phi: np.ndarray, psi: np.ndarray, omega: np.ndarray) -> np.ndarray:",
    ),
    (
        "def pos(p1, p2, p3, bl, ba, di):",
        "def pos(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, bl: float, ba: float, di: float) -> np.ndarray:",
    ),
    (
        "def _write_simple_pdb(coords, res_names, path):",
        "def _write_simple_pdb(coords: np.ndarray, res_names: List[str], path: str) -> None:",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# quality/gnn/model.py
# ─────────────────────────────────────────────────────────────────────────────
PATCHES["quality/gnn/model.py"] = [
    (
        "def _check_pyg():",
        "def _check_pyg() -> None:",
    ),
    (
        "def __init__(self):\n        super().__init__()",
        "def __init__(self) -> None:\n        super().__init__()",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Also: validator.py validate_sequence_improbabilities and validate_steric_clashes
# need return type on the closing paren (they're multi-line)
# ─────────────────────────────────────────────────────────────────────────────
MULTILINE_PATCHES: dict[str, list[tuple[str, str]]] = {
    "validator.py": [
        # validate_steric_clashes — add return type at end of closing paren
        (
            "        min_ca_distance: float = 3.0,\n    ):",
            "        min_ca_distance: float = 3.0,\n    ) -> Dict[str, Any]:",
        ),
        # validate_sequence_improbabilities
        (
            "        max_hydrophobic_stretch: int = 10,\n    ):",
            "        max_hydrophobic_stretch: int = 10,\n    ) -> Dict[str, Any]:",
        ),
    ],
    "torsion.py": [
        # export_torsion_angles — add return type at closing paren
        (
            "    fmt: str = 'csv',\n):",
            "    fmt: str = 'csv',\n) -> None:",
        ),
    ],
    "distogram.py": [
        # export_distogram
        (
            "    fmt: str = 'csv',\n):",
            "    fmt: str = 'csv',\n) -> None:",
        ),
    ],
}


def ensure_typing_imports(text: str, path: Path) -> str:
    """Make sure Any, Union, Optional, Tuple, List, Dict are imported where needed."""
    needed = set()
    if (
        "Any" in text and "Any" not in text.split("from typing import")[1].split("\n")[0]
        if "from typing import" in text
        else True
    ):
        needed.add("Any")
    # Just check if Union is now referenced but not imported
    for name in ["Any", "Union", "Tuple"]:
        if f": {name}" in text or f"[{name}" in text or f"{name}," in text:
            needed.add(name)

    if not needed or "from typing import" not in text:
        return text

    # Find existing typing import line and augment it
    import re

    m = re.search(r"from typing import ([^\n]+)", text)
    if m:
        existing = [x.strip() for x in m.group(1).split(",")]
        new_ones = [n for n in sorted(needed) if n not in existing]
        if new_ones:
            new_import = "from typing import " + ", ".join(sorted(set(existing + new_ones)))
            text = text[: m.start()] + new_import + text[m.end() :]
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Apply all patches
# ─────────────────────────────────────────────────────────────────────────────
total_applied = 0
total_skipped = 0

for rel_path, patches in list(PATCHES.items()) + list(MULTILINE_PATCHES.items()):
    path = ROOT / rel_path
    if not path.exists():
        print(f"SKIP (not found): {rel_path}")
        continue
    text = path.read_text(encoding="utf-8")
    applied = 0
    for old, new in patches:
        if old in text:
            text = text.replace(old, new, 1)
            applied += 1
        else:
            print(f"  WARN [{rel_path}]: not found: {old[:60]!r}")
            total_skipped += 1
    if applied:
        text = ensure_typing_imports(text, path)
        try:
            ast.parse(text)
        except SyntaxError as e:
            print(f"  SYNTAX ERROR in {rel_path}: {e}")
            continue
        path.write_text(text, encoding="utf-8")
        total_applied += applied
        print(f"  {rel_path}: {applied} annotations added")

print(f"\nTotal annotations added: {total_applied}  |  Skipped (not found): {total_skipped}")
