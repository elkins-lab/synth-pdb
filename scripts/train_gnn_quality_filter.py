#!/usr/bin/env python3

from __future__ import annotations

"""Train the GNN-based protein quality classifier.

Trains the ProteinGNN with two jointly optimised objectives:

1. **Global binary classification** (Good / Bad) — NLL loss on graph-pooled output.
2. **Per-residue pLDDT regression** — MSE loss on per-node sigmoid output vs.
   Ramachandran Z-score targets derived from ground-truth torsion angles.

─────────────────────────────────────────────────────────────────────────────
STRUCTURAL BIOLOGY CONTEXT — Why this multi-task approach?
─────────────────────────────────────────────────────────────────────────────

A structural biologist typically judges protein quality using two lenses:
  • Global fold: Does the overall architecture look like a protein?
  • Local geometry: Are the bond lengths, angles, and torsions physically
    plausible (e.g., Ramachandran favoured, no steric clashes)?

Standard GNN classifiers often focus on the global fold but ignore local
"micro-violations" like a single clashing atom. By adding an auxiliary
per-residue task (pLDDT regression), we force the GNN's internal message-passing
layers to encode local geometric features (φ/ψ angles).

pLDDT (predicted Local Distance Difference Test) is the same metric used by
AlphaFold 2 to communicate confidence. Here, we use a Ramachandran-derived
Z-score as a ground-truth proxy for pLDDT during training.

Usage
-----
    python scripts/train_gnn_quality_filter.py
    python scripts/train_gnn_quality_filter.py --n-samples 400 --epochs 100
    python scripts/train_gnn_quality_filter.py --n-samples 40 --epochs 5 --output /tmp/test.pt
"""

import argparse
import logging
import math
import os
import sys
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Per-residue pLDDT target generation
# ─────────────────────────────────────────────────────────────────────────────

# Ideal Ramachandran centres for alpha-helix (φ=-60°, ψ=-45°) and
# beta-strand (φ=-120°, ψ=+120°).  Used to derive per-residue quality targets.
# These centres represent the "sweet spots" of the backbone energy landscape
# where steric hindrance between side-chain atoms and the backbone is minimised.
_HELIX_PHI, _HELIX_PSI = -60.0, -45.0
_STRAND_PHI, _STRAND_PSI = -120.0, 120.0
_COIL_PHI, _COIL_PSI = -60.0, 140.0  # PPII/extended approximate centre


def _ramachandran_score(phi: float | None, psi: float | None) -> float:
    """Return a Ramachandran-based per-residue quality score ∈ [0, 1].

    Educational note — Ramachandran Z-score and Steric Constraints
    ────────────────────────────────────────────────────────────────
    A residue's backbone torsion angles (φ, ψ) are not free to rotate 360°.
    The peptide bond has a planar character, and the Cβ atom of the side-chain
    collides with the backbone O and N atoms if the angles are "forbidden."

    We compute the distance from the nearest "favoured" centre (alpha, beta,
    or poly-proline II) and convert it to a score via a Gaussian kernel
    with σ=40°.

    • Residue in a perfect alpha-helix → score ≈ 1.0 (High confidence).
    • Residue in a "disallowed" region → score ≈ 0.0 (Unphysical/Clashing).

    This is a simplified but physically motivated proxy for MolProbity's
    per-residue Ramachandran Z-score used in CASP (Critical Assessment
    of Structure Prediction) quality assessment.
    """
    if phi is None or psi is None:
        # Terminal residues have undefined dihedrals (missing C_{i-1} or N_{i+1})
        # We assign a moderate confidence of 0.5.
        return 0.5

    sigma = 40.0  # degrees — approximate half-width of favoured regions

    def gaussian_dist(phi_ref: float, psi_ref: float) -> float:
        dphi = phi - phi_ref
        dpsi = psi - psi_ref
        # Handle wrap-around for circular dihedrals
        if dphi > 180:
            dphi -= 360
        if dphi < -180:
            dphi += 360
        if dpsi > 180:
            dpsi -= 360
        if dpsi < -180:
            dpsi += 360
        d2 = dphi**2 + dpsi**2
        return math.exp(-d2 / (2 * sigma**2))

    # Score = max Gaussian score across the three main favoured regions.
    # We take the maximum so a residue in ANY favoured region scores high.
    score = max(
        gaussian_dist(_HELIX_PHI, _HELIX_PSI),
        gaussian_dist(_STRAND_PHI, _STRAND_PSI),
        gaussian_dist(_COIL_PHI, _COIL_PSI),
    )
    return float(score)


def compute_per_residue_targets(
    pdb_content: str,
    label: int,
    perturbed_residue_idx: int | None = None,
) -> np.ndarray:
    """Compute per-residue pLDDT targets for training.

    pLDDT (predicted Local Distance Difference Test) is a per-residue confidence
    metric. In our GNN, we want the model to predict high pLDDT for well-formed
    residues and low pLDDT for those with distorted geometry.

    Args:
        pdb_content: PDB-format string.
        label: Global structure label (1 = Good, 0 = Bad).
        perturbed_residue_idx: If this structure was distorted by perturbing a
            specific residue (e.g. a clash injection), pass its index here.
            That residue receives a hard target of 0.0 (High error) to teach
            the GNN to spot local steric violations even if torsions are okay.

    Returns:
        np.ndarray of shape [N,] with per-residue targets ∈ [0, 1].
    """
    from synth_pdb.quality.gnn.graph import _parse_backbone

    residues = _parse_backbone(pdb_content)
    targets = np.array(
        [_ramachandran_score(r["phi"], r["psi"]) for r in residues],
        dtype=np.float32,
    )

    # For Bad structures produced by clash injection, the perturbed residue
    # should explicitly receive low confidence even if its torsions look ok.
    if label == 0 and perturbed_residue_idx is not None:
        idx = int(perturbed_residue_idx)
        if 0 <= idx < len(targets):
            targets[idx] = 0.0

    return targets


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_pdb_dataset(n_samples: int = 200, random_state: int = 42, diverse_good: bool = False):
    """Generate synthetic PDB strings across four structural classes.

    Why these four classes?
    ───────────────────────
    1. GOOD: Provides a baseline for idealized, "folded" geometry.
       Includes physical 'wobble' (drift 0-15°) to ensure robustness.
    2. RANDOM (Coil): Models "disordered" or "unfolded" states with unconstrained
       torsions. Teaches the GNN to recognise forbidden Ramachandran regions.
    3. DISTORTED/DRIFTED: Models marginal "Bad" structures.
       - 1/3 are 'shaken' (0.5 Å noise).
       - 1/3 are 'drunken' (High drift 40-60°). This defines the quality cliff.
       - 1/3 are 'surgical' (Perfect backbone with 1-2 random residues).
         This teaches the GNN that "One bad apple spoils the bunch."
    4. CLASHING: Models "threading" or "sampling" errors where 1-4 residues
       are physically in the wrong place (steric clash).

    Args:
        n_samples: Total samples to generate.
        random_state: RNG seed.
        diverse_good: If True, the "Good" category includes Beta and PPII
            conformations in addition to Alpha helices.

    Returns
    -------
    pdbs   : list[str]   — PDB format strings
    labels : np.ndarray  — global label (1 = Good, 0 = Bad)
    clash_indices : list[int | None]
        For Clashing structures, a representative residue index that was displaced.
    """
    import io

    import biotite.structure.io.pdb as pdb_io

    from synth_pdb.generator import generate_pdb_content

    rng = np.random.default_rng(random_state)

    n_good = int(n_samples * 0.4)
    n_bad_random = int(n_samples * 0.2)
    n_bad_distorted = int(n_samples * 0.2)
    n_bad_clash = n_samples - n_good - n_bad_random - n_bad_distorted

    logger.info(
        "Generating: %d Good%s, %d Random, %d Distorted/Drifted/Surgical, %d Clashing",
        n_good,
        " (Diverse)" if diverse_good else "",
        n_bad_random,
        n_bad_distorted,
        n_bad_clash,
    )

    pdbs: list[str] = []
    labels: list[int] = []
    clash_indices: list[int | None] = []
    failure_counts = {"good": 0, "random": 0, "distorted": 0, "clash": 0}

    # 1. Good — idealized backbone geometry with physical 'wobble'
    for i in range(n_good):
        if i % 20 == 0:
            logger.info("  Good %d/%d", i, n_good)
        try:
            conf = "alpha"
            if diverse_good:
                conf = rng.choice(["alpha", "beta", "ppii"], p=[0.6, 0.3, 0.1])

            # Small drift (0-15°) is still considered 'Good'.
            drift_val = rng.uniform(0.0, 15.0)

            pdbs.append(
                generate_pdb_content(
                    length=20, conformation=conf, drift=drift_val, minimize_energy=False
                )
            )
            labels.append(1)
            clash_indices.append(None)
        except Exception as e:
            failure_counts["good"] += 1
            logger.warning("Good sample %d failed: %s", i, e)

    # 2. Bad (Random coil — poor Ramachandran geometry)
    for i in range(n_bad_random):
        if i % 10 == 0:
            logger.info("  Random %d/%d", i, n_bad_random)
        try:
            pdbs.append(
                generate_pdb_content(length=20, conformation="random", minimize_energy=False)
            )
            labels.append(0)
            clash_indices.append(None)
        except Exception as e:
            failure_counts["random"] += 1
            logger.warning("Random sample %d failed: %s", i, e)

    # 3. Bad (Distorted/Drifted/Surgical — defining the cliff)
    for i in range(n_bad_distorted):
        if i % 10 == 0:
            logger.info("  Distorted/Drifted/Surgical %d/%d", i, n_bad_distorted)
        try:
            rval = rng.random()
            if rval < 0.33:
                # Type A: Coordinate noise (0.5 Å)
                clean = generate_pdb_content(length=20, conformation="alpha", minimize_energy=False)
                f = io.StringIO(clean)
                struc_obj = pdb_io.PDBFile.read(f).get_structure(model=1)
                struc_obj.coord += rng.normal(0, 0.5, struc_obj.coord.shape)
                f_out = io.StringIO()
                pdb_file = pdb_io.PDBFile()
                pdb_file.set_structure(struc_obj)
                pdb_file.write(f_out)
                pdbs.append(f_out.getvalue())
            elif rval < 0.66:
                # Type B: High Torsion Drift (40-60°) — Marginal "Bad"
                pdbs.append(
                    generate_pdb_content(
                        length=20,
                        conformation=rng.choice(["alpha", "beta"]),
                        drift=rng.uniform(40.0, 60.0),
                        minimize_energy=False,
                    )
                )
            else:
                # Type C: Surgical Torsion Corruption (1-2 bad residues in good backbone)
                # This forces the GNN to be sensitive to local "broken" spots.
                n_corrupt = rng.integers(1, 3)
                # Generate indices like "5-5:random" or "5-6:random"
                start = rng.integers(2, 18)
                end = start + n_corrupt - 1
                struct_spec = f"{start}-{end}:random"
                pdbs.append(
                    generate_pdb_content(
                        length=20,
                        structure=struct_spec,
                        conformation=rng.choice(["alpha", "beta"]),
                        minimize_energy=False,
                    )
                )
            labels.append(0)
            clash_indices.append(None)
        except Exception as e:
            failure_counts["distorted"] += 1
            logger.warning("Distorted sample %d failed: %s", i, e)

    # 4. Bad (Clashing — multi-residue displacement)
    for i in range(n_bad_clash):
        if i % 10 == 0:
            logger.info("  Clashing %d/%d", i, n_bad_clash)
        try:
            clean = generate_pdb_content(length=20, conformation="alpha", minimize_energy=False)
            f = io.StringIO(clean)
            struc_obj = pdb_io.PDBFile.read(f).get_structure(model=1)
            ca_idx = [j for j, a in enumerate(struc_obj) if a.atom_name == "CA"]

            # Perturb 1 to 4 random residues
            n_perturb = rng.integers(1, 5)
            perturbed_indices = rng.choice(len(ca_idx), size=n_perturb, replace=False)

            for p_idx in perturbed_indices:
                target_idx = (p_idx + 5) % len(ca_idx)
                struc_obj.coord[ca_idx[p_idx]] = struc_obj.coord[ca_idx[target_idx]]

            f_out = io.StringIO()
            pdb_file = pdb_io.PDBFile()
            pdb_file.set_structure(struc_obj)
            pdb_file.write(f_out)
            pdbs.append(f_out.getvalue())
            labels.append(0)
            clash_indices.append(int(perturbed_indices[0]))
        except Exception as e:
            failure_counts["clash"] += 1
            logger.warning("Clash sample %d failed: %s", i, e)

    total_failures = sum(failure_counts.values())
    if total_failures:
        logger.warning("Generation failures: %s (total=%d)", failure_counts, total_failures)

    return pdbs, np.array(labels, dtype=np.int64), clash_indices


def build_graph_dataset(y: np.ndarray, pdb_list: list, clash_indices: list):
    """Convert raw PDB strings to PyG Data objects."""
    import torch
    from synth_pdb.quality.gnn.graph import build_protein_graph

    graphs = []
    for pdb_content, label, clash_idx in zip(pdb_list, y, clash_indices):
        try:
            g = build_protein_graph(pdb_content)
            g.y = torch.tensor([int(label)], dtype=torch.long)
            targets = compute_per_residue_targets(pdb_content, int(label), clash_idx)
            g.residue_targets = torch.tensor(targets, dtype=torch.float)
            graphs.append(g)
        except Exception as e:
            logger.debug("Graph build failed: %s", e)
    return graphs


def train_gnn(
    output: str,
    n_samples: int = 200,
    epochs: int = 50,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    random_state: int = 42,
    residue_loss_weight: float = 0.3,
    diverse_good: bool = False,
):
    """Train the GNN quality classifier with joint global + per-residue loss."""
    try:
        import torch
        import torch.nn.functional as F
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
        from torch_geometric.loader import DataLoader
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        sys.exit(1)

    from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier
    from synth_pdb.quality.gnn.model import ProteinGNN

    pdbs, y, clash_indices = generate_pdb_dataset(
        n_samples=n_samples, random_state=random_state, diverse_good=diverse_good
    )
    graphs = build_graph_dataset(y, pdbs, clash_indices)
    if not graphs:
        raise RuntimeError("No graphs built.")

    y_graphs = np.array([g.y.item() for g in graphs])
    idx = np.arange(len(graphs))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_graphs)
    train_graphs = [graphs[i] for i in idx_train]
    test_graphs = [graphs[i] for i in idx_test]

    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinGNN(node_features=8, edge_features=2, hidden_dim=hidden_dim, num_classes=2).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info("Training Robust GNN (λ_residue=%.2f)...", residue_loss_weight)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            log_probs, per_residue_scores = type(model).forward_with_node_embeddings(
                model, batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            global_loss = F.nll_loss(log_probs, batch.y)
            residue_loss = F.mse_loss(per_residue_scores.squeeze(-1), batch.residue_targets)
            loss = global_loss + residue_loss_weight * residue_loss
            loss.backward()
            optimizer.step()
            total_loss += global_loss.item() * batch.num_graphs
            total += batch.num_graphs
        scheduler.step()
        if epoch % max(1, epochs // 10) == 0:
            logger.info("Epoch %3d/%d  global_loss=%.4f", epoch, epochs, total_loss / total)

    # ── Evaluation ─────────────────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            log_probs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds = log_probs.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch.y.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    logger.info("\n=== Test Evaluation ===")
    logger.info("Accuracy: %.4f", acc)
    logger.info(
        "\n%s",
        classification_report(all_labels, all_preds, target_names=["Bad", "Good"], labels=[0, 1]),
    )

    clf = GNNQualityClassifier.__new__(GNNQualityClassifier)
    clf.model = model.cpu()
    clf._model_path = None
    clf.save(output)
    logger.info("Saved Robust model to %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="synth_pdb/quality/models/gnn_quality_v2.pt")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--residue-loss-weight", type=float, default=0.3)
    parser.add_argument("--diverse-good", action="store_true")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    train_gnn(**vars(args))
