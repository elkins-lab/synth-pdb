# 🚀 synth-pdb Incubator: Frontier Explorations

This document serves as a roadmap and brainstorming space for experimental, high-impact use cases for `synth-pdb`. The goal is to push the boundaries of synthetic protein generation into areas where traditional structural biology (X-ray, NMR) reaches its limits.

---

## 🏗️ 1. The Cryo-EM "Standard Candle" Simulator
**The Vision:** Generate atomic-resolution synthetic density maps for massive complexes to benchmark Cryo-EM refinement software.

*   **The Scientific Gap:** Cryo-EM is often limited by "local resolution" issues. We lack a "perfect" ground truth for how a 1.2 Å map should look for complex proteins like Apoferritin when subjected to specific noise profiles.
*   **The `synth-pdb` Angle:** We can generate thousands of slightly varied "thermal states" of a protein at tensor-speed, which can be used to simulate the "blurring" seen in Cryo-EM maps.
*   **Current Status:** 🧪 Prototype Stage (Need to integrate map-to-density conversion tools).

---

## ☁️ 2. IDP "Ensemble-First" Validation
**The Vision:** Create the first automated pipeline for generating and validating "Conformational Ensembles" for Intrinsically Disordered Proteins (IDPs) like Alpha-synuclein.

*   **The Scientific Gap:** IDPs are "invisible" to X-ray and Cryo-EM. NMR provides data, but interpreting that data requires a massive ensemble of potential shapes that is computationally expensive to generate.
*   **The `synth-pdb` Angle:** Our `BatchedGenerator` can create 10,000+ physically plausible random-coil conformations in seconds, providing the raw material for ensemble-averaging calculations (PRE, RDC, SAXS).
*   **Current Status:** 🛠️ Active Development (See `examples/interactive_tutorials/idp_ensemble_validation.ipynb`).

---

## 🌌 3. Mapping the "Dark Proteome" (AI Benchmarking)
**The Vision:** Use `synth-pdb` to generate "Hard Decoys" for AI-predicted structures that have no experimental solution.

*   **The Scientific Gap:** AlphaFold-3 and ESM-Fold have predicted millions of structures, but we don't know where they might be "hallucinating" (e.g., in flexible loops or orphan proteins).
*   **The `synth-pdb` Angle:** We can generate "near-native" decoys—structures that look like the AI prediction but have subtle, physically plausible changes—to test if the AI's confidence scores (pLDDT) can actually distinguish them.
*   **Current Status:** 💡 Concept Stage.

---

## 🧪 4. De Novo Miniprotein Forge
**The Vision:** A rapid prototyping environment for synthetic biologists to "visualize" their custom-designed sequences before they ever hit the lab.

*   **The Scientific Gap:** Designing small, stable peptides (miniproteins) often requires heavy Rosetta simulations.
*   **The `synth-pdb` Angle:** `synth-pdb` provides a "lightweight" alternative to quickly generate and check the geometry (bond angles, clashes) of a new design.
*   **Current Status:** 💡 Concept Stage.

---

## 💊 5. Pharmacophore Clash Filter for Peptide Drug Discovery
**The Vision:** Use `synth-pdb`'s distogram and clash scoring to rapidly pre-filter peptide drug candidates against a known binding pocket before any docking simulation.

*   **The Scientific Gap:** Physics-based docking (AutoDock, Glide) is expensive — typically 10–60 minutes per compound. Millions of peptide candidates cannot be screened this way. A fast geometric pre-filter that eliminates obvious clashes would dramatically shrink the search space.
*   **The `synth-pdb` Angle:** `scoring.calculate_clash_score()` and `distogram.calculate_distogram()` are already implemented. Combined with the `docking` module's `write_pqr()` helper, we could build a tiered funnel: geometric clash → distogram similarity to known binders → full docking. The `drug_discovery_pipeline.ipynb` tutorial covers late-stage docking but skips this fast pre-filter step entirely.
*   **Key Insight:** For peptide drugs, the dominant failure mode is steric exclusion from the binding site — exactly what clash scoring detects cheaply.
*   **Graduation Path:** Extend `drug_discovery_pipeline.ipynb` with a "pre-screen" section, then graduate to a standalone notebook.
*   **Current Status:** 💡 Concept Stage. All required building blocks (`scoring`, `distogram`, `docking`) are implemented.

---

## 🗺️ 6. Contact Map Fingerprinting for Fold Classification
**The Vision:** Generate a large library of synthetic structures with known folds (α, β, α/β) and train a lightweight classifier on their contact maps — creating a fast "fold fingerprint" that categorises unknown structures without sequence alignment.

*   **The Scientific Gap:** Sequence-based fold classification (SCOP, CATH) breaks down in the "twilight zone" (<30% identity). Structural classifiers (DALI, TM-align) are slow. A contact-map-based approach sits in the middle: structure-derived, but significantly faster.
*   **The `synth-pdb` Angle:** `contact.compute_contact_map()` is implemented and returns a binary matrix. `BatchedGenerator` can produce thousands of labelled structures (labelled by the `--structure` string used to generate them) in seconds. The contact matrices are naturally 2D arrays — perfect input for a CNN or simple SVM classifier. This is a clean end-to-end ML story: generate → featurise → classify.
*   **Graduation Path:** Working prototype graduates to `examples/ml_integration/` alongside the distogram and orientogram notebooks.
*   **Current Status:** 🧪 Prototype Stage. `contact.py` and `BatchedGenerator` are ready; no classifier code exists yet.

---

## 🧬 7. Synthetic SAXS Curve Generator
**The Vision:** Compute synthetic Small-Angle X-ray Scattering (SAXS) curves from generated structures to provide ground-truth data for benchmarking SAXS analysis software (ATSAS, BioXTAS RAW).

*   **The Scientific Gap:** SAXS is the only technique that measures IDPs and large complexes in solution at near-physiological conditions — but software validation requires structures whose SAXS curve is "known". Experimental SAXS always has instrument noise; synthetic curves from exact coordinates would be perfectly clean.
*   **The `synth-pdb` Angle:** A SAXS curve is computed from inter-atomic distances using the Debye formula, which is a direct function of `distogram.calculate_distogram()`. The IDP ensemble machinery already computes pairwise distances. Adding the Debye summation is a tractable extension of existing code with no new dependencies.
*   **Connects To:** Incubator ideas #2 (IDP ensembles) and #1 (Cryo-EM standard candles) — SAXS curves are used alongside both techniques for cross-validation.
*   **Current Status:** 💡 Concept Stage. Blocked only on implementing the Debye equation summation (~50 lines of NumPy).

---

## 🔬 8. GFP Chromophore Engineering Sandbox
**The Vision:** Extend the existing GFP chromophore implementation into an interactive engineering environment — mutate residues around the chromophore, regenerate the structure, and predict spectral shifts using geometric proxies.

*   **The Scientific Gap:** Fluorescent protein engineering requires iterative mutagenesis to shift emission wavelength. Full computational prediction requires quantum chemistry (TD-DFT). However, geometric proxies — chromophore planarity, H-bond partner count, cavity burial depth — are known spectral correlates that can be computed quickly from structure.
*   **The `synth-pdb` Angle:** `special_chemistry.find_gfp_chromophore_motif()` and `form_gfp_chromophore()` are implemented. `packing.optimize_sidechains()` can repack the local environment after mutation. `calculate_residue_sasa()` can estimate chromophore burial depth. Combining these gives a lightweight "spectral proxy" without quantum chemistry.
*   **Connects To:** The existing `gfp_molecular_forge.ipynb` tutorial — this would be a direct advanced extension of that notebook at the Advanced tier.
*   **Current Status:** 🧪 Prototype Stage. Core chemistry primitives are ready; the mutagenesis loop and spectral proxy scoring are not yet implemented.

---

## 🧩 9. Co-evolutionary Fitness Landscape Explorer
**The Vision:** Combine MSA generation (`evolution`, `msa`) with SASA burial and rotamer quality to build a "fitness landscape" heat map showing which single-point mutations are structurally tolerated vs. destabilising.

*   **The Scientific Gap:** Deep Mutational Scanning (DMS) experiments measure mutational fitness but require large-scale wet lab work. Computational landscapes (EVmutation, ESM-1v) exist but are opaque black boxes with no structural interpretability.
*   **The `synth-pdb` Angle:** `evolution.generate_msa_sequences()` provides sequence variation. `burial_ratio` + rotamer quality from `get_quality_report()` provide structural scoring. The workflow: generate all single-point mutants → score each with the defensibility scorecard → visualise as a 2D heat map (position × amino acid). This is transparent and directly connects sequence to structural consequence.
*   **Scalability Note:** With `BatchedGenerator`, scanning 20 mutations × 50 positions = 1,000 structures takes seconds, making this genuinely interactive.
*   **Connects To:** The `protein_quality_assessment.ipynb` scorecard added in this session (Step 6) would be the core scoring primitive for this explorer.
*   **Current Status:** 💡 Concept Stage. All components are individually implemented; no integration code exists yet.

---

## 🏥 10. Synthetic Ensemble for NMR Order Parameter Validation
**The Vision:** Generate ensembles of physically plausible backbone conformations with known conformational diversity and use them as synthetic ground truth to validate Model-Free (Lipari-Szabo) analysis software.

*   **The Scientific Gap:** Software packages like `FAST`, `relax`, and `Modelfree` need test cases with known S² values to validate their fitting routines. Real NMR data is noisy; a synthetic ensemble with controlled S² (via known RMSF) would be a perfect benchmark with a provably correct answer.
*   **The `synth-pdb` Angle:** `relaxation.predict_order_parameters()` is implemented. `BatchedGenerator` can produce ensembles with tunable conformational diversity (via `drift` and `rmsd_max`). A calibration loop — vary input diversity → compute S² per residue → plot vs. drift magnitude — gives a transparent S² ground-truth curve.
*   **Connects To:** Directly extends `virtual_nmr_spectrometer.ipynb` and `alphafold_vs_nmr_dynamics.ipynb`, providing the scientific validation backbone that both currently lack.
*   **Current Status:** 💡 Concept Stage.

---

## 📝 How to Contribute an Exploration
1.  **Draft a Vision**: Define the "What If?".
2.  **Define the Gap**: Why can't traditional tools do this easily?
3.  **Code a Prototype**: Add a script in a subfolder within `/incubator/`.
4.  **Validate & Graduate**: If the idea proves stable and useful, it "graduates" to the core `examples/` directory.
