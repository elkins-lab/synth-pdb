#!/usr/bin/env python3

from __future__ import annotations

"""
Deep Comparison: Alpha-only vs Diverse GNN Quality Scorer.

This script performs a rigorous comparison between two trained models:
1. Baseline: Trained only on Alpha helices (and various 'Bad' types).
2. Diverse: Trained on Alpha, Beta, and PPII (and various 'Bad' types).

SCIENTIFIC OBJECTIVE:
To quantify the "generalisation gap" in GNN-based quality filters. We evaluate
how a model trained only on helices fails when encountering beta-strands or
PPII linkers, and how the --diverse-good flag fixes this bias.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

# Ensure repo root and scripts dir are in path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = str(_REPO_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier
from train_gnn_quality_filter import generate_pdb_dataset, build_graph_dataset

logger = logging.getLogger(__name__)


def evaluate_on_conformation(model, graphs, name):
    """Evaluate accuracy and pLDDT error for a specific structural group."""
    model.model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)

    all_preds = []
    all_labels = []
    all_plddt_errs = []

    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            # Use the dual-output head
            log_probs, per_residue_scores = type(model.model).forward_with_node_embeddings(
                model.model, g.x, g.edge_index, g.edge_attr, g.batch
            )

            preds = log_probs.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(g.y.cpu().numpy().tolist())

            # Per-residue pLDDT MSE
            mse = torch.mean((per_residue_scores.squeeze(-1) - g.residue_targets) ** 2).item()
            all_plddt_errs.append(mse)

    acc = accuracy_score(all_labels, all_preds)
    mean_plddt_mse = np.mean(all_plddt_errs)

    return {"accuracy": acc, "plddt_mse": mean_plddt_mse, "count": len(graphs)}


def deep_compare(baseline_path, diverse_path, n_test=100):
    """Run cross-conformation benchmarks."""
    logger.info("Loading models...")
    baseline = GNNQualityClassifier(baseline_path)
    diverse = GNNQualityClassifier(diverse_path)

    # ── Test Set Generation ──────────────────────────────────────────
    # We generate a large test set that includes ALL types.
    # 1. Pure Alpha-Good
    # 2. Pure Beta-Good
    # 3. Pure PPII-Good
    # 4. Bad (Random/Distorted/Clash)

    groups = {
        "Alpha-Good": {"conf": "alpha", "label": 1},
        "Beta-Good": {"conf": "beta", "label": 1},
        "PPII-Good": {"conf": "ppii", "label": 1},
        "Random-Bad": {"conf": "random", "label": 0},
    }

    from synth_pdb.generator import generate_pdb_content
    from train_gnn_quality_filter import compute_per_residue_targets

    results = {}

    for group_name, cfg in groups.items():
        logger.info(f"Generating test group: {group_name}...")
        pdbs = []
        labels = []
        for _ in range(n_test):
            try:
                p = generate_pdb_content(length=20, conformation=cfg["conf"], minimize_energy=False)
                pdbs.append(p)
                labels.append(cfg["label"])
            except:
                pass

        # Build graphs
        graphs = build_graph_dataset(np.array(labels), pdbs, [None] * len(pdbs))

        # Evaluate both models
        res_baseline = evaluate_on_conformation(baseline, graphs, "Baseline")
        res_diverse = evaluate_on_conformation(diverse, graphs, "Diverse")

        results[group_name] = {"baseline": res_baseline, "diverse": res_diverse}

    # ── Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'CONFORMATION GROUP':<15} | {'BASELINE ACC':<12} | {'DIVERSE ACC':<12} | {'DELTA':<8}")
    print("-" * 80)

    for group, data in results.items():
        b_acc = data["baseline"]["accuracy"]
        d_acc = data["diverse"]["accuracy"]
        delta = d_acc - b_acc
        print(f"{group:<15} | {b_acc:12.2%} | {d_acc:12.2%} | {delta:+.1%}")

    print("=" * 80)
    print("\nEducational Insight:")
    print("If Baseline Accuracy is high on Alpha but low on Beta/PPII, it proves the model")
    print("was 'overfit' to a single structural motif. The Diverse model should show")
    print("balanced performance across all physically valid geometries.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Deep comparison of GNN Quality Scorer models")
    parser.add_argument("--baseline", required=True, help="Path to alpha-only model (.pt)")
    parser.add_argument("--diverse", required=True, help="Path to diverse-good model (.pt)")
    parser.add_argument("--n-test", type=int, default=50, help="Samples per group")
    args = parser.parse_args()

    deep_compare(args.baseline, args.diverse, n_test=args.n_test)
