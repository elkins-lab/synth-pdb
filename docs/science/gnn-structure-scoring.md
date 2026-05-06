# GNN Structure Scoring — Scientific Background

> *"A protein structure is only as good as the evidence used to build it.  
> Computational scoring lets us quantify that evidence — one residue at a time."*

---

## Why Do We Need Automated Quality Scoring?

Protein structure determination has traditionally been validated by human experts
using tools like MolProbity, PROCHECK, and WhatCheck. These tools excel for
experimentally-determined structures, but the explosion of **AI-predicted structures**
(AlphaFold2, ESMFold, RoseTTAFold) has created a new challenge: hundreds of
millions of predicted structures, each needing rapid quality assessment.

synth-pdb addresses this with a **Graph Attention Network (GNN)** that scores
structures in milliseconds and provides per-residue confidence outputs analogous
to AlphaFold's pLDDT score.

---

## From Ramachandran to pLDDT: A Brief History of Quality Metrics

### The Ramachandran Plot (1963)

The foundational quality metric in structural biology. The backbone dihedral angles
**φ (phi)** and **ψ (psi)** are plotted for each residue. Only certain (φ, ψ) combinations
are sterically allowed:

```
        ψ
   180 ┤
       │         ████████████
       │         ████████████  ← β-strand region
       │         ████████████  (φ ≈ −120°, ψ ≈ +120°)
     0 ┼─────────────────────
       │    ███████████
       │    ███████████        ← α-helix region
       │    ███████████        (φ ≈ −60°, ψ ≈ −45°)
  −180 ┤
        └────────────────────
      −180         0       180
                            φ
```

**MolProbity standard** (Richardson lab, Duke University):

| Quality   | Favoured | Outliers |
|-----------|----------|----------|
| Excellent | > 98%    | < 0.2%   |
| Good      | > 95%    | < 2%     |
| Marginal  | > 90%    | < 5%     |
| Poor      | < 90%    | > 5%     |

### B-factors / Crystallographic Temperature Factors

In X-ray crystallography, the **B-factor** (also called the temperature factor or
Debye-Waller factor) quantifies the spread of electron density around each atom.

$$B = 8\pi^2 \langle u^2 \rangle$$

where $\langle u^2 \rangle$ is the mean-square displacement. A high B-factor
(typically > 80 Å²) indicates the atom is mobile or the density is poorly
resolved — and the local structure should be treated with caution.

### AlphaFold's pLDDT (2021)

AlphaFold2 introduced a per-residue confidence metric called **pLDDT**
(predicted Local Distance Difference Test), which predicts how well the model
would score on the lDDT metric if compared to an experimental structure.

**pLDDT colour coding** (as displayed in the AlphaFold Database):

| pLDDT   | Colour | Interpretation                          |
|---------|--------|------------------------------------------|
| > 90    | Blue   | Very high confidence — likely accurate  |
| 70–90   | Cyan   | High confidence                         |
| 50–70   | Yellow | Low confidence — treat with caution     |
| < 50    | Orange | Very low confidence — likely disordered |

!!! note "pLDDT ≠ experimental B-factor"
    pLDDT is a model confidence metric, not a physical temperature factor.
    A residue with pLDDT = 40 may be genuinely disordered *or* may simply
    be in a region where the model lacked sufficient evolutionary information.
    Always consider the biological context.

synth-pdb's GNN quality scorer produces an analogous per-residue confidence
score, using the **same four-band colour scheme** for direct comparability.

---

## What is a Graph Neural Network?

### The Protein as a Graph

In structural biology, proteins are naturally represented as graphs:

- **Nodes** = residues (one node per amino acid)
- **Edges** = contacts between residues within 8 Å Cα–Cα distance

This representation captures the **contact topology** — the network of
interactions that define a protein fold — rather than the raw Cartesian
coordinates, which depend on the orientation of the molecule in space.

```
   Residues as nodes:          Contact graph:
   
   Ala-1 ── Gly-2              1 ─── 2
   |                           │   / │
   contact (i+4)               │  /  │
   |                           4 ─── 3
   Val-4 ── Ser-3
```

### Node Features: What Each Residue "Knows"

Each node in the graph carries an **8-dimensional feature vector** encoding
the local backbone geometry of that residue:

| Feature         | Meaning                                    | Range        |
|-----------------|--------------------------------------------|--------------|
| `sin_phi`       | Sine of backbone dihedral φ                | [−1, +1]     |
| `cos_phi`       | Cosine of φ (together with sin, encodes angle without wraparound) | [−1, +1] |
| `sin_psi`       | Sine of backbone dihedral ψ                | [−1, +1]     |
| `cos_psi`       | Cosine of ψ                                | [−1, +1]     |
| `b_factor_norm` | Normalised crystallographic B-factor       | [0, 1]       |
| `seq_position`  | Position in chain (0 = N-terminus, 1 = C-terminus) | [0, 1] |
| `is_n_terminus` | 1 if N-terminal residue                    | {0, 1}       |
| `is_c_terminus` | 1 if C-terminal residue                    | {0, 1}       |

!!! tip "Why sin/cos encoding?"
    Angles are **circular** — 179° and −179° are only 2° apart, but their
    raw values differ by 358°. Encoding angle θ as (sin θ, cos θ) gives a
    continuous, distance-preserving 2D representation: the Euclidean
    distance between two encoded angles equals the chord length on the unit
    circle, which is monotonically related to the angular difference.

### Edge Features: What Contacts "Know"

Each edge carries a **2-dimensional feature vector**:

| Feature              | Meaning                                          |
|----------------------|--------------------------------------------------|
| Cα–Cα distance       | Distance between the two residues (Å)            |
| Sequence separation  | \|i − j\| — distinguishes local (peptide bond) from long-range contacts |

### Message Passing: How the GNN Learns

A GNN learns by **message passing**: each node iteratively collects information
from its neighbours and updates its own representation.

After **layer 1**, each residue knows about its immediate neighbours (i±1, i±2, i±3...).  
After **layer 2**, each residue knows about neighbours-of-neighbours.  
After **layer 3**, information has propagated 3 hops — covering most of a small helix.

The **Graph Attention Network (GAT)** variant used here assigns **learned attention
weights** to each edge, so the model can learn to emphasise, say, a long-range
contact that causes a steric clash, while down-weighting routine peptide-bond edges.

---

## Multi-Task Learning: Global + Per-Residue Outputs

The synth-pdb GNN is trained with **two simultaneous objectives**:

```
                         Node embeddings [N × 64]
                              ↙              ↘
              Global pooling              Per-residue head
              (mean over N)               (MLP per node)
                    ↓                           ↓
            Graph embedding              Per-residue scores
              [batch × 64]                  [N × 1]
                    ↓                           ↓
           P(Good / Bad)              pLDDT ∈ [0, 1] per residue
              (binary)                  (regression target)
```

**Why multi-task?**  

1. **Efficiency** — one forward pass produces both outputs
2. **Regularisation** — the per-residue auxiliary task forces the shared message-passing
   backbone to encode local geometry, which *also* improves global classification performance

This is analogous to how AlphaFold2 predicts pLDDT during training as an auxiliary
head alongside the main structure prediction objective.

---

## Benchmark Metrics Explained

### TM-score

**Template Modelling score** — the primary metric in CASP (Critical Assessment of
Structure Prediction) competitions since 2004.

$$\text{TM} = \frac{1}{L_{\text{ref}}} \sum_{i=1}^{N_{\text{aligned}}} \frac{1}{1 + \left(\frac{d_i}{d_0}\right)^2}$$

where:

- $d_i$ = distance between aligned Cα pair $i$ after optimal superposition
- $L_{\text{ref}}$ = length of the reference chain
- $d_0 = 1.24 (L_{\text{ref}} - 15)^{1/3} - 1.8$ Å — a length-normalising constant that makes TM-score length-independent

| TM-score | Interpretation                                    |
|----------|---------------------------------------------------|
| > 0.9    | Near-atomic accuracy (better than most crystal forms) |
| > 0.5    | Same global fold (by definition of "same fold")  |
| 0.17     | Random — expected value for two unrelated structures |
| < 0.17   | Worse than random (physically impossible topology) |

!!! important "TM-score vs RMSD"
    RMSD is sensitive to a few large local deviations (e.g. one disordered loop).
    TM-score is more robust: it asks "does the overall fold match?" rather than
    "are all atoms in the same place?". This makes TM-score the preferred global
    accuracy metric for structure prediction benchmarking.

### lDDT — Local Distance Difference Test

**Per-residue metric, no superposition required.**

For each residue $i$, we look at all residue pairs $(i, j)$ where $j$ is
within 15 Å in the reference structure. We then ask: what fraction of those
reference inter-residue distances are reproduced in the prediction to within
{0.5, 1, 2, 4} Å?

$$\text{lDDT}_i = \frac{1}{4} \sum_{t \in \{0.5, 1, 2, 4\}} \frac{|\{j : |d^{\text{pred}}_{ij} - d^{\text{ref}}_{ij}| < t\}|}{|\{j : d^{\text{ref}}_{ij} < 15\text{ Å}\}|}$$

**Key advantage**: lDDT doesn't require superposition, so it is not confused by
global rotations or translations. It measures *local* structural accuracy.

AlphaFold's pLDDT is a **prediction of lDDT** — not lDDT itself. The model
predicts how well it thinks it will score, and these self-assessments correlate
strongly with actual lDDT when evaluated against experimental structures.

### GDT-TS — Global Distance Test, Total Score

**The standard CASP ranking metric** since CASP4 (2000).

$$\text{GDT-TS} = \frac{1}{4}\left(f_{1\text{ Å}} + f_{2\text{ Å}} + f_{4\text{ Å}} + f_{8\text{ Å}}\right)$$

where $f_{d\text{ Å}}$ is the fraction of Cα atoms placed within $d$ Å of the
reference after optimal superposition.

GDT-TS = 1.0 means every Cα is within 1 Å of the reference (excellent).  
GDT-TS ≈ 0.3 is typical for homology models based on distant templates.

### Kabsch Superposition

All superposition-based metrics (TM-score, GDT-TS, RMSD) require first aligning
the two structures in 3D space. The **Kabsch algorithm** finds the optimal rotation
matrix via Singular Value Decomposition (SVD):

1. Translate both structures to their centroids
2. Compute the cross-covariance matrix $H = \mathbf{M}^\top \mathbf{R}$
3. Decompose: $H = U \Sigma V^\top$
4. Rotation: $R = V \,\text{diag}(1, 1, \det(VU^\top))\, U^\top$
5. The det term prevents reflections (ensures a proper rotation)

### Chemical Shift RMSD

Chemical shifts are exquisitely sensitive reporters of local backbone geometry.
The **shift RMSD** between two structures is computed as a weighted average over
nuclei ($^1$H, $^{13}$C, $^{15}$N):

$$\text{shift RMSD} = \sqrt{\frac{\sum_n w_n \sum_i (\delta_i^{\text{pred}} - \delta_i^{\text{ref}})^2}{\sum_n w_n N_n}}$$

Default weights follow the SPARTA+ convention:

| Nucleus | Weight | Rationale                              |
|---------|--------|----------------------------------------|
| $^1$H   | 1.00   | Reference; most sensitive to geometry  |
| $^{13}$C| 0.25   | Broader chemical shift range           |
| $^{15}$N| 0.10   | Less sensitive to local geometry       |

A shift RMSD < 0.5 ppm ($^1$H) is generally considered excellent agreement.

---

## When to Use Which Metric

| Situation | Best metric | Why |
|-----------|-------------|-----|
| Comparing AI predictions to experiment | TM-score | Global fold accuracy, length-independent |
| Finding locally wrong regions | lDDT (per-residue) | No superposition, residue-level detail |
| CASP-style ranking | GDT-TS | Historical standard, enables comparison with literature |
| NMR structure validation | Shift RMSD | Chemical shifts report on backbone geometry directly |
| Quick quality filter | GNN pLDDT | Fast (< 1 ms), correlates with Ramachandran quality |
| Publication submission | Ramachandran % | Required by journals and PDB deposition |

---

## Further Reading

- **Zhang & Skolnick (2004)** — Original TM-score paper. *Proteins*, 57, 702–710.
- **Mariani et al. (2013)** — lDDT metric definition. *Bioinformatics*, 29, 2722–2728.
- **Jumper et al. (2021)** — AlphaFold2, including pLDDT. *Nature*, 596, 583–589.
- **Chen et al. (2010)** — MolProbity. *Acta Cryst D*, 66, 12–21.
- **Ramachandran et al. (1963)** — The original Ramachandran plot. *JMB*, 7, 95–99.

---

*See also:*

- [API Reference: score_structure()](../api/scoring.md)
- [API Reference: benchmark metrics](../api/benchmark.md)
- [Tutorial: GNN pLDDT Explorer](../tutorials/gnn_plddt_explorer.ipynb)
- [Tutorial: Protein Quality Assessment](../tutorials/protein_quality_assessment.ipynb)
