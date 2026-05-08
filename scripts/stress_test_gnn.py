#!/usr/bin/env python3

from __future__ import annotations

"""
GNN Quality Scorer Stress Test: Searching for the Breaking Point.

SCIENTIFIC OBJECTIVE:
To find the exact geometric threshold where the GNN stops calling a structure
'Good'. We test sensitivity to:
1.  Torsion Drift: How much 'wobble' can an alpha helix take before it's 'Bad'?
2.  Local Corruption: Can a single forbidden residue break a global score?
3.  Mixed motifs: How do hybrid Alpha-Beta chains score?
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure repo root and scripts dir are in path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = str(_REPO_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier
from train_gnn_quality_filter import build_graph_dataset, compute_per_residue_targets
from synth_pdb.generator import generate_pdb_content

logger = logging.getLogger(__name__)


def predict_score(model, pdb_content):
    """Get the 'Good' probability (0-1) for a PDB string."""
    # build_graph_dataset expects a list/array of labels
    graphs = build_graph_dataset(np.array([1]), [pdb_content], [None])
    if not graphs:
        return 0.0

    model.model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)

    with torch.no_grad():
        g = graphs[0].to(device)
        # Global output (log_probs)
        log_probs = model.model(g.x, g.edge_index, g.edge_attr, g.batch)
        probs = torch.exp(log_probs)
        return probs[0, 1].item()  # Index 1 is 'Good'


def stress_test_torsion_drift(model, n_steps=10, samples_per_step=20):
    """Sweep noise levels on an alpha helix backbone."""
    print("\n[STRESS TEST 1] Torsion Drift (Backbone Wobble)")
    print("Goal: Find the noise level (degrees) where the model breaks.")

    noise_levels = np.linspace(0, 60, n_steps)
    mean_scores = []

    for sigma in noise_levels:
        scores = []
        for _ in range(samples_per_step):
            # The 'drift' parameter in our generator adds Gaussian noise to phi/psi
            pdb = generate_pdb_content(
                length=20, conformation="alpha", drift=sigma, minimize_energy=False
            )
            scores.append(predict_score(model, pdb))

        mean_score = np.mean(scores)
        mean_scores.append(mean_score)
        print(f"  Noise σ = {sigma:4.1f}° | Mean 'Good' Prob: {mean_score:6.2%}")

    return noise_levels, mean_scores


def stress_test_local_corruption(model):
    """Corrupt just one residue in a perfect helix."""
    print("\n[STRESS TEST 2] Local Corruption (One Bad Apple)")
    print("Goal: Can a single unphysical residue trigger a 'Bad' global label?")

    # 1. Perfect Helix
    clean_pdb = generate_pdb_content(length=20, conformation="alpha", minimize_energy=False)
    clean_score = predict_score(model, clean_pdb)
    print(f"  Perfect Helix Score: {clean_score:.2%}")

    # 2. Helix with one 'forbidden' residue
    # We sample a random coil, but for the GNN classifier, we can't easily
    # surgically edit the PDB string here without a parser.
    # Instead, we'll use the 'structure' feature of the generator to inject
    # a 'random' residue into an 'alpha' chain.
    corrupt_pdb = generate_pdb_content(
        length=20,
        structure="10-10:random",  # Only residue 10 is random/bad
        conformation="alpha",
        minimize_energy=False,
    )
    corrupt_score = predict_score(model, corrupt_pdb)
    print(f"  One Bad Residue Score: {corrupt_score:.2%}")

    # 3. Two Bad Residues
    corrupt_2_pdb = generate_pdb_content(
        length=20, structure="10-11:random", conformation="alpha", minimize_energy=False
    )
    corrupt_2_score = predict_score(model, corrupt_2_pdb)
    print(f"  Two Bad Residues Score: {corrupt_2_score:.2%}")


def run_suite(model_path):
    logger.info(f"Loading model from {model_path}...")
    model = GNNQualityClassifier(model_path)

    noise, scores = stress_test_torsion_drift(model)
    stress_test_local_corruption(model)

    # ── Interpret ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STRESS TEST ANALYSIS")
    print("=" * 80)

    # Find the 'Critical σ' where score drops below 50%
    critical_sigma = None
    for n, s in zip(noise, scores):
        if s < 0.5:
            critical_sigma = n
            break

    if critical_sigma:
        print(f"CRITICAL DRIFT: The model breaks at ~{critical_sigma:.1f}° of torsion noise.")
        if critical_sigma < 20:
            print("Verdict: ULTRA-SENSITIVE. (Strict physics)")
        elif critical_sigma < 40:
            print("Verdict: BALANCED. (Follows Ramachandran favoured boundaries)")
        else:
            print("Verdict: PERMISSIVE. (Only catches extreme garbage)")
    else:
        print("CRITICAL DRIFT: Never found. The model is too permissive!")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN Quality Scorer Stress Test")
    parser.add_argument("--model", required=True, help="Path to the model to test (.pt)")
    args = parser.parse_args()

    run_suite(args.model)
