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


def generate_pdb_dataset(n_samples: int = 200, random_state: int = 42):
    """Generate synthetic PDB strings across four structural classes.

    Why these four classes?
    ───────────────────────
    1. GOOD (Alpha Helix): Provides a baseline for idealized, "folded"
       geometry with perfect H-bonding and Ramachandran placement.
    2. RANDOM (Coil): Models "disordered" or "unfolded" states with unconstrained
       torsions. Teaches the GNN to recognise forbidden Ramachandran regions.
    3. DISTORTED: Models "shaken" structures or low-resolution models where
       the global fold is roughly correct but local Cα positions have 0.5 Å
       noise. Teaches the GNN to be sensitive to fine-grained coordinate error.
    4. CLASHING: Models "threading" or "sampling" errors where a single atom
       is physically in the wrong place (steric clash). Teaches the GNN to
       spot local outliers in an otherwise good structure.

    Returns
    -------
    pdbs   : list[str]   — PDB format strings
    labels : np.ndarray  — global label (1 = Good, 0 = Bad)
    clash_indices : list[int | None]
        For Clashing structures, the residue index that was displaced.
        None for all other structure types (used for per-residue target generation).
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
        "Generating: %d Good, %d Random, %d Distorted, %d Clashing",
        n_good,
        n_bad_random,
        n_bad_distorted,
        n_bad_clash,
    )

    pdbs: list[str] = []
    labels: list[int] = []
    clash_indices: list[int | None] = []
    failure_counts = {"good": 0, "random": 0, "distorted": 0, "clash": 0}

    # 1. Good (Alpha Helix) — idealized backbone geometry
    for i in range(n_good):
        if i % 20 == 0:
            logger.info("  Good %d/%d", i, n_good)
        try:
            pdbs.append(
                generate_pdb_content(length=20, conformation="alpha", minimize_energy=False)
            )
            labels.append(1)
            clash_indices.append(None)
        except Exception as e:
            failure_counts["good"] += 1
            logger.warning("Good sample %d failed: %s", i, e, exc_info=True)

    # 2. Bad (Random coil — poor Ramachandran geometry)
    # Torsion angles are sampled uniformly [0, 360], ignoring steric hindrance.
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
            logger.warning("Random sample %d failed: %s", i, e, exc_info=True)

    # 3. Bad (Distorted — good helix with Gaussian coordinate noise)
    # This simulates a "loose" structure where Cα atoms are wobbling away
    # from their ideal lattice positions.
    for i in range(n_bad_distorted):
        if i % 10 == 0:
            logger.info("  Distorted %d/%d", i, n_bad_distorted)
        try:
            clean = generate_pdb_content(length=20, conformation="alpha", minimize_energy=False)
            f = io.StringIO(clean)
            struc_obj = pdb_io.PDBFile.read(f).get_structure(model=1)
            struc_obj.coord += rng.normal(0, 0.5, struc_obj.coord.shape)
            f_out = io.StringIO()
            pdb_file = pdb_io.PDBFile()
            pdb_file.set_structure(struc_obj)
            pdb_file.write(f_out)
            pdbs.append(f_out.getvalue())
            labels.append(0)
            clash_indices.append(None)
        except Exception as e:
            failure_counts["distorted"] += 1
            logger.warning("Distorted sample %d failed: %s", i, e, exc_info=True)

    # 4. Bad (Clashing — single Cα displacement)
    # We move a Cα atom directly on top of another residue's position.
    # This creates a massive steric clash (Van der Waals overlap) which is
    # a classic sign of a poor-quality structural model.
    for i in range(n_bad_clash):
        if i % 10 == 0:
            logger.info("  Clashing %d/%d", i, n_bad_clash)
        try:
            clean = generate_pdb_content(length=20, conformation="alpha", minimize_energy=False)
            f = io.StringIO(clean)
            struc_obj = pdb_io.PDBFile.read(f).get_structure(model=1)
            ca_idx = [j for j, a in enumerate(struc_obj) if a.atom_name == "CA"]
            clash_res_idx = None
            if len(ca_idx) >= 5:
                struc_obj.coord[ca_idx[1]] = struc_obj.coord[ca_idx[4]]
                clash_res_idx = 1  # residue index 1 (0-based) was displaced
            f_out = io.StringIO()
            pdb_file = pdb_io.PDBFile()
            pdb_file.set_structure(struc_obj)
            pdb_file.write(f_out)
            pdbs.append(f_out.getvalue())
            labels.append(0)
            clash_indices.append(clash_res_idx)
        except Exception as e:
            failure_counts["clash"] += 1
            logger.warning("Clash sample %d failed: %s", i, e, exc_info=True)

    total_failures = sum(failure_counts.values())
    if total_failures:
        logger.warning("Generation failures: %s (total=%d)", failure_counts, total_failures)
    else:
        logger.info("All %d samples generated successfully.", len(pdbs))

    if len(pdbs) < int(n_samples * 0.5):
        raise RuntimeError(
            f"Only {len(pdbs)} of {n_samples} samples generated. Failures: {failure_counts}"
        )

    return pdbs, np.array(labels, dtype=np.int64), clash_indices


def build_graph_dataset(y: np.ndarray, pdb_list: list, clash_indices: list):
    """Convert raw PDB strings to PyG Data objects with global labels and per-residue targets.

    Each protein is represented as a contact graph:
      Nodes: Residues (Cα atoms)
      Edges: Connectivity if Cα–Cα distance < 8 Å.

    Args:
        y: Global label array (1 = Good, 0 = Bad), length N.
        pdb_list: List of N PDB strings.
        clash_indices: List of per-structure clash residue indices (or None).

    Returns:
        List of torch_geometric.data.Data objects with .y and .residue_targets set.
    """
    import torch

    from synth_pdb.quality.gnn.graph import build_protein_graph

    graphs = []
    failures = 0
    for i, (pdb_content, label, clash_idx) in enumerate(
        zip(pdb_list, y, clash_indices, strict=False)
    ):
        try:
            g = build_protein_graph(pdb_content)
            g.y = torch.tensor([int(label)], dtype=torch.long)

            # Per-residue pLDDT targets based on local geometry
            targets = compute_per_residue_targets(pdb_content, int(label), clash_idx)
            g.residue_targets = torch.tensor(targets, dtype=torch.float)  # [N,]

            graphs.append(g)
        except Exception as e:
            failures += 1
            logger.warning("Graph build failed for sample %d: %s", i, e)

    if failures:
        logger.warning("%d / %d graph builds failed.", failures, len(pdb_list))
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


def train_gnn(
    output_path: str,
    n_samples: int = 200,
    epochs: int = 50,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    random_state: int = 42,
    residue_loss_weight: float = 0.3,
):
    """Train the GNN quality classifier with joint global + per-residue loss.

    The model learns to predict:
    1. Is this structure valid overall? (Global Label)
    2. Which residues have bad local geometry? (Per-residue pLDDT)

    This multi-task setup regularises the GNN, making it more robust than
    a simple binary classifier.

    Args:
        output_path: Where to save the .pt checkpoint.
        n_samples: Training samples to generate.
        epochs: Number of training epochs.
        hidden_dim: GNN hidden dimension.
        lr: Initial learning rate (AdamW + CosineAnnealing).
        random_state: RNG seed for reproducibility.
        residue_loss_weight: Weight λ for the per-residue MSE loss term.
            Total loss = NLL_global + λ × MSE_per_residue.
            Default 0.3 balances the two tasks without overwhelming the
            primary classification objective.
    """
    try:
        import torch
        import torch.nn.functional as F
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
        from torch_geometric.loader import DataLoader
    except ImportError as exc:
        logger.error("Missing dependency: %s. Run: pip install synth-pdb[gnn]", exc)
        sys.exit(1)

    from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier
    from synth_pdb.quality.gnn.model import ProteinGNN

    # ── Data generation ────────────────────────────────────────────────
    logger.info("=== GNN Quality Scorer Training (v2 — with per-residue pLDDT head) ===")
    pdbs, y, clash_indices = generate_pdb_dataset(n_samples=n_samples, random_state=random_state)

    logger.info("Building protein graphs with per-residue targets...")
    graphs = build_graph_dataset(y, pdbs, clash_indices)

    if not graphs:
        raise RuntimeError("No graphs were built. Aborting training.")

    y_graphs = np.array([g.y.item() for g in graphs])

    idx = np.arange(len(graphs))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_graphs)
    train_graphs = [graphs[i] for i in idx_train]
    test_graphs = [graphs[i] for i in idx_test]

    logger.info(
        "Dataset: %d train / %d test  (Good: %d, Bad: %d)",
        len(train_graphs),
        len(test_graphs),
        int(np.sum(y_graphs == 1)),
        int(np.sum(y_graphs == 0)),
    )

    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

    # ── Model, optimiser, scheduler ────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)

    model = ProteinGNN(node_features=8, edge_features=2, hidden_dim=hidden_dim, num_classes=2)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ──────────────────────────────────────────────────
    # λ_residue (residue_loss_weight) controls how much the model prioritises
    # local geometry (pLDDT) over global classification.
    logger.info("Training for %d epochs (λ_residue=%.2f)...", epochs, residue_loss_weight)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Dual-output forward pass.
            # We call via type(model) to bypass nn.Module.__getattr__, which
            # intercepts attribute lookup on Module instances and only checks
            # _parameters / _buffers / _modules — it never reaches ordinary
            # methods defined on inner classes created by the __new__ factory.
            log_probs, per_residue_scores = type(model).forward_with_node_embeddings(
                model, batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )

            # ── Loss 1: global binary classification ───────────────────
            global_loss = F.nll_loss(log_probs, batch.y)

            # ── Loss 2: per-residue pLDDT regression ──────────────────
            # Forces the GNN to understand local residue quality.
            # residue_targets is [total_nodes,] concatenated from all graphs
            # in the batch (same ordering as batch.x rows).
            residue_loss = F.mse_loss(
                per_residue_scores.squeeze(-1),  # [total_nodes]
                batch.residue_targets,  # [total_nodes]
            )

            # ── Combined loss ─────────────────────────────────────────
            loss = global_loss + residue_loss_weight * residue_loss
            loss.backward()
            optimizer.step()

            total_loss += global_loss.item() * batch.num_graphs
            preds = log_probs.argmax(dim=-1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs

        scheduler.step()
        train_acc = correct / total if total > 0 else 0.0

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            logger.info(
                "Epoch %3d/%d  global_loss=%.4f  train_acc=%.3f  lr=%.2e",
                epoch,
                epochs,
                total_loss / total,
                train_acc,
                scheduler.get_last_lr()[0],
            )

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

    # ── Save ───────────────────────────────────────────────────────────
    clf = GNNQualityClassifier.__new__(GNNQualityClassifier)
    clf.model = model.cpu()
    clf._model_path = None
    clf.save(output_path)
    logger.info("Done! Model saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Train GNN Protein Quality Classifier (Graph Attention Network)"
    )
    parser.add_argument(
        "--output",
        default="synth_pdb/quality/models/gnn_quality_v2.pt",
        help="Output path for the .pt checkpoint",
    )
    parser.add_argument("--n-samples", type=int, default=200, help="Training samples to generate")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GNN hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--random-state", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--residue-loss-weight",
        type=float,
        default=0.3,
        help="Weight λ for per-residue MSE loss (default 0.3)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    train_gnn(
        output_path=args.output,
        n_samples=args.n_samples,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        random_state=args.random_state,
        residue_loss_weight=args.residue_loss_weight,
    )
