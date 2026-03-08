# Tutorial Suggestions: Interactive & Visual NMR Content

> This file tracks proposed interactive Jupyter Notebook tutorials for visually explaining
> NMR concepts and `synth-pdb` / `synth-nmr` functionality. Prioritised by educational gap
> and implementation effort.

---

## What Already Exists

### synth-pdb: `examples/interactive_tutorials/virtual_nmr_spectrometer.ipynb`
The flagship interactive tutorial. Covers:
- ✅ `ipywidgets` sliders for field strength and tumbling time (R₁/R₂/NOE update live)
- ✅ `py3Dmol` 3D structure viewer coloured by B-factor / rigidity
- ✅ 2D ¹H–¹⁵N HSQC scatter plot
- ✅ Chemical Shift Index (CSI) bar chart
- ✅ J-coupling bar chart (Karplus equation)
- ✅ Contact map / distogram
- ❌ No RDC content at all
- ❌ No `--shift-predictor` comparison

### synth-nmr: `docs/tutorials/advanced_observables.ipynb`
Covers J-couplings, NOEs, and RDCs on Ubiquitin (1D3Z), but:
- ⚠️ The RDC section is **entirely static** — one frozen bar chart, no interactivity
- ⚠️ No visual explanation of what the alignment tensor is (no polar map)
- ⚠️ No comparison of Da and R values

---

## Priority 1 — Interactive Alignment Tensor Explorer ✅ DONE

**File**: `examples/interactive_tutorials/rdc_alignment_explorer.ipynb`

Fills the biggest educational gap. Covers:
1. **Polar RDC map** — 3D colour map on unit sphere showing D(θ,φ); sliders for Da and R
2. **Live per-residue RDC bar chart** — connected to `synth_pdb.rdc.calculate_rdcs`
3. **Alignment media comparison panel** — bicelle, phage Pf1, gels with literature Da/R
4. **RDC vs NOE complementarity panel** — head-to-head local vs global restraints

---

## Priority 2 — Add RDC Section to `virtual_nmr_spectrometer.ipynb`

**File**: `examples/interactive_tutorials/virtual_nmr_spectrometer.ipynb`

- Add **"RDC Fingerprint"** section with Da/R slider pair → bar chart via `calculate_rdcs`
- Add **3D colour-by-RDC view** in py3Dmol (red = positive, blue = negative)

---

## Priority 3 — Shift Predictor Side-by-Side Comparison ✅ DONE

**File**: `examples/interactive_tutorials/virtual_nmr_spectrometer.ipynb`

- Side-by-side HSQC scatter plots: Empirical (left) vs SHIFTX2 (right) — with graceful fallback if SHIFTX2 not installed
- Per-residue Δδ(¹H) bar chart (SHIFTX2 − empirical), color-coded by sign
- Summary stats table (std dev, range, RMSD) with literature RMSD benchmarks
- References: Han et al. 2011 *J Biomol NMR* 50:43; Shen &amp; Bax 2010 *J Biomol NMR* 48:13

---

## Priority 4 — Q-Factor RDC Validation Panel

**File**: `examples/interactive_tutorials/protein_quality_assessment.ipynb` (extend)

- Compare RDCs from two structures (refined vs coil)
- Show Q-factor agreement score (Cornilescu et al., 1998, *J Biomol NMR* 13:289)
- Teaches: low Q = good structure agreement with experimental RDCs

---

## Implementation Timeline Estimate

| # | Notebook | Effort | Impact |
|---|---|---|---|
| 1 | New `rdc_alignment_explorer.ipynb` | ~4 hrs | Very high |
| 2 | Extend `virtual_nmr_spectrometer.ipynb` | ~2 hrs | High |
| 3 | Shift predictor comparison | ~1 hr | Medium |
| 4 | Q-factor in `protein_quality_assessment.ipynb` | ~2 hrs | Medium |
