# Synth-PDB: Long-Term AI Integration Strategy

To ensure `synth-pdb` remains innovative, competitive, and highly useful in the rapidly evolving
landscape of computational structural biology, the integration of new AI features must be strategic.

The core philosophy should be **"Physics + AI"**. Rather than competing directly with monolithic
models like AlphaFold 3, `synth-pdb` should position itself as the ultimate **AI Data Factory,
Validation Engine, and Surrogate Simulator**.

Here is a deep-dive roadmap into features that would solidify `synth-pdb`'s reputation as a
cutting-edge tool.

---

## 1. Generative Backbone & Loop Sampling (Diffusion / Flow Matching)

Currently, `synth-pdb` relies on the NeRF algorithm and Ramachandran probability distributions
for random coils.

*   **The Feature:** Integrate a lightweight, pre-trained Diffusion Model or Flow Matching model
    specifically for generating contiguous backbones conditionally. Conditioning signals could
    include a secondary structure string (e.g., `HHHHEEEE`), a sparse NOE distance restraint set,
    a partial distance map, or a text description of the desired topology.
*   **Proposed Models:** FrameDiff, Genie2, or a custom lightweight model trained on synthetic
    `synth-pdb` trajectories themselves — a compelling feedback loop.
*   **Missing Detail — "Inpainting" Use Case:** The most commercially valuable sub-feature is
    *loop inpainting*: a user provides a rigid scaffold (e.g., two alpha-helices) and the model
    samples a physically realistic connecting loop. This is a critical need in antibody CDR loop
    modelling and enzyme active-site design.
*   **Missing Detail — Conditional Generation from NMR Restraints:** Allow users to provide a set
    of sparse NOE distance bounds as conditioning signals. The diffusion model would then sample
    backbones that satisfy those restraints, directly bridging experimental NMR data with
    computational structure generation — a genuinely novel capability.
*   **Why it's competitive:** It bridges the gap between pure random sampling and rigorous physics.
    Users could "inpaint" missing loops or generate completely novel miniprotein topologies in
    milliseconds, bypassing the need for heavy external tools like RFdiffusion for simple tasks.

---

## 2. Integrated Inverse Folding (Sequence Design)

Currently, `synth-pdb` generates structure from sequence (or random sequences).

*   **The Feature:** Incorporate an Inverse Folding model (similar to ProteinMPNN or ESM-IF)
    directly into the pipeline.
*   **Workflow:** A user specifies a backbone geometry (e.g., an ideal TIM barrel or a synthetic
    macrocycle scaffold), and the AI auto-designs a sequence that stably folds into that shape.
    The designed sequence is then re-validated by the existing OpenMM physics engine and the
    `quality` GNN scorer to close the loop.
*   **Missing Detail — Multi-State Design:** Extend inverse folding to target *conformational
    ensembles*, not just single structures. Design sequences that adopt conformation A in condition
    X (e.g., ligand-bound) and conformation B in condition Y (e.g., apo). This is the holy grail
    of allosteric protein design and is not achievable by any current open-source tool.
*   **Missing Detail — Negative Design:** Explicitly penalize sequences that would fold into
    off-target structures. This is essential for therapeutic peptides where selectivity matters.
    The existing `decoys.py` hard-decoy engine can be repurposed to supply the negative examples
    for this training objective.
*   **Why it's useful:** It transforms `synth-pdb` from a "generator" into a **"De Novo Design
    Forge"**, highly attractive to synthetic biologists and drug designers.

---

## 3. Adversarial Decoy Generation (GANs / RL)

Currently, decoys are generated via sequence threading or torsion drift in `decoys.py`.

*   **The Feature:** Train a Generative Adversarial Network (GAN) or a Reinforcement Learning (RL)
    agent to generate "Hard Decoys" specifically designed to fool state-of-the-art predictors like
    AlphaFold and ESMFold. The OpenMM classical energy and the existing `quality` GNN can serve
    as the reward signal — decoys must appear energetically favorable while being topologically
    impossible.
*   **Missing Detail — RL Reward Shaping:** Use a multi-term reward: minimize OpenMM energy (looks
    physical), maximize GNN pLDDT confidence (fools the scorer), while simultaneously maximizing
    knottedness or a topology violation metric. This three-way adversarial tension produces the
    hardest possible negative samples.
*   **Missing Detail — Benchmark Integration:** Publish a "Decoy Challenge Leaderboard" where
    users submit their structure prediction models to be tested against `synth-pdb` hard decoys.
    This would make the community *depend* on `synth-pdb` as an external evaluation harness.
*   **Why it's innovative:** The AI field desperately needs high-quality negative data. If
    `synth-pdb` becomes the gold standard for "realistic but fundamentally flawed" structures,
    every major AI lab will incorporate it into their training pipelines.

---

## 4. Differentiable Physics Integration (JAX-MD / TorchMD)

Currently, `synth-pdb` relies on OpenMM for classical minimization.

*   **The Feature:** Provide an optional, modular, fully auto-differentiable physics backend
    (e.g., using JAX-MD or TorchMD-Net).
*   **Missing Detail — Gradient-Through-Physics Workflow:** The key use case is enabling a user to
    define a loss function (e.g., predicted vs. experimental SAXS curve), run a forward pass through
    the `synth-pdb` physics engine, and backpropagate gradients all the way into the initial
    torsion angles or model weights. This makes `synth-pdb` a native, trainable *layer* in a deep
    learning architecture — not just a pre-processing tool.
*   **Missing Detail — Energy-as-a-Loss:** Expose OpenMM potential energy as a differentiable
    scalar loss via the TorchMD-Net `EnergyModel`. This allows generative models (diffusion,
    inverse folding) to be fine-tuned on-the-fly so that their outputs satisfy physical energy
    constraints without a separate relaxation step.
*   **Why it's competitive:** This is the holy grail for AI researchers. It makes `synth-pdb` a
    native layer in deep learning architectures, enabling end-to-end training from sequence all
    the way to experimental observables.

---

## 5. AI Surrogates for Spectroscopy (NMR & Cryo-EM)

Currently, `synth-pdb` uses empirical rules (SPARTA-lite) for chemical shifts and mathematical
approximations for RDCs/SAXS. A `cryo_em.py` module already exists as a foundation.

*   **The Feature:** Train and bundle highly optimized Neural Network Surrogates — specifically
    Graph Neural Networks (GNNs) operating on the molecular graph — that predict NMR parameters
    (chemical shifts, J-couplings, RDCs, relaxation rates T1/T2) and simulate Cryo-EM density
    maps (with realistic noise, CTF, and water boxes) in milliseconds.
*   **Missing Detail — NMR Shift Predictor (ShiftML-style):** Replace the SPARTA-lite empirical
    formula with a trained GNN surrogate. The model takes the local atomic environment of each
    residue (coordinates, bond graph, neighbors within 8 Å) and outputs ¹H, ¹³C, and ¹⁵N
    chemical shifts with near-DFT accuracy. The existing `chemical_shifts.py` data would serve
    as ground-truth training labels.
*   **Missing Detail — Relaxation Rate Surrogate:** The existing `relaxation.py` module uses the
    Lipari-Szabo model. A GNN surrogate trained on MD simulation data could predict `S²` order
    parameters and `τc` correlation times directly from structure, enabling instant NMR dynamics
    prediction without MD.
*   **Missing Detail — Cryo-EM Density Map Synthesis:** The existing `cryo_em.py` module is the
    right foundation. Extend it with a learned forward model that applies realistic CTF, ice
    contamination noise, and preferred orientation artifacts — all learnable parameters — to
    create *adversarially realistic* synthetic Cryo-EM datasets for training 2D-to-3D
    reconstruction networks.
*   **Why it's innovative:** Neural surrogates would allow users to generate thousands of labeled
    `(Structure, Spectrum)` pairs per second, enabling the training of "Spectroscopy-to-Structure"
    AI models that are currently starved for training data.

---

## 6. Generative Ensembles for Intrinsically Disordered Proteins (IDPs)

The existing `ensemble` subpackage provides a foundation for multi-structure analysis.

*   **The Feature:** Instead of generating single static structures or random walks, deploy a
    Generative Flow Network (GFlowNet) constrained by polymer physics priors (Flory scaling,
    Kratky-Porod worm-like chain) and trained on the Protein Ensemble Database (PED) to generate
    statistically accurate, multi-state ensembles of IDPs.
*   **Missing Detail — SAXS/SANS Reweighting (BME/EOM):** Integrate Bayesian/Maximum Entropy (BME)
    reweighting directly into the ensemble generator. The user provides a measured SAXS or SANS
    curve; the tool generates a raw ensemble and then reweights the population of structures to
    maximize agreement with the experimental data. This is the standard protocol used in
    publications (EOM, EROS, BME tools), but no single library currently makes it
    end-to-end and scriptable.
*   **Missing Detail — Phase Separation (LLPS) Predictor:** Train a classifier on sequence
    features (charge pattern, aromatic content, hydropathy) to predict propensity for
    liquid-liquid phase separation (LLPS). LLPS is a $1B+ research topic in neurodegenerative
    disease (TDP-43, FUS, hnRNPA1). Adding this feature would make `synth-pdb` immediately
    relevant to a massive and well-funded research community.
*   **Why it's useful:** IDPs are a massive blind spot for current AI (including AlphaFold).
    Providing a tool that accurately models conformational landscapes of disordered regions with
    biophysical realism would capture a massive academic and pharmaceutical audience.

---

## 7. Fast Neural Scoring & Quality Filtering (GNNs)

The `quality/gnn` subpackage already contains `model.py`, `graph.py`, and `gnn_classifier.py`,
making this the most implementation-ready item on the roadmap.

*   **The Feature:** Deploy and expose the existing `quality` GNN as a fully documented,
    high-throughput Model Quality Assessment (MQA) scoring function. It should score stability,
    identify steric clashes, predict pLDDT-like confidence metrics, and output an overall
    "naturalness" score in milliseconds per structure.
*   **Missing Detail — pLDDT-Like Per-Residue Confidence:** Extend the GNN output head to produce
    per-residue confidence scores, not just a global score. This allows users to identify exactly
    which loops or termini are low-quality, not just that the overall structure is poor. This
    mirrors the most useful output of AlphaFold.
*   **Missing Detail — Active Learning Loop:** Use the GNN scorer as the oracle in an active
    learning pipeline: generate N structures → score with GNN → select top-K for OpenMM
    relaxation → use relaxed energies to retrain the GNN. This self-improving loop would
    continuously increase scoring accuracy with no additional human labeling.
*   **Missing Detail — Pre-Trained Weights Distribution:** Bundle pre-trained GNN weights with
    the package (via Git LFS or HuggingFace Hub), so the scorer works out-of-the-box without
    requiring users to train their own model. This is the single biggest usability barrier.
*   **Why it's competitive:** Acts as a massive high-throughput filter. Users can generate
    100,000 rough decoys, use the GNN to filter to the top 1%, and only run expensive classical
    physics on the most promising candidates — a 100× speedup for screening pipelines.

---

## 8. LLM-Driven Agents & Multimodal Orchestration

The `llm.py` module already implements `LocalLLMProvider` and `OpenAILLMProvider` with JSON
schema-constrained output and Strategy pattern backends.

*   **The Feature:** Evolve the existing LLM integration from a "prompt-to-CLI-args" translator
    into a true *structural agent*. The agent should be able to decompose multi-step tasks
    ("design a zinc-finger peptide, validate it, and export the NMR training dataset") into a
    sequence of `synth-pdb` API calls, execute them, and report back in natural language.
*   **Missing Detail — Tool-Calling Architecture:** Implement the agent as a formal tool-calling
    LLM loop (ReAct / function-calling pattern). Each `synth-pdb` public API function becomes a
    "tool" the LLM can invoke. The agent can chain: `generate → score → minimize → generate_shifts
    → export_npz`, reasoning at each step about whether the output is acceptable.
*   **Missing Detail — Multimodal Report Generation:** Allow the agent to generate a complete
    natural-language biophysical report from a PDB file, synthesizing information from the
    validator, chemical shift predictor, GNN scorer, and SAXS simulator into a coherent narrative.
    This would be enormously useful for teaching and for non-expert users.
*   **Missing Detail — Structured Knowledge Base (RAG):** Augment the LLM with a Retrieval-
    Augmented Generation (RAG) layer over the `synth-pdb` documentation and BMRB/PDB metadata.
    Users could ask "what are typical ¹H chemical shifts for a buried Trp in an alpha helix?"
    and receive scientifically accurate answers grounded in real data.
*   **Why it's useful:** Lowers the barrier to entry to zero and enables `synth-pdb` to
    autonomously generate massive `(Text, Structure, Spectrum)` multimodal datasets for training
    the next generation of biological foundation models.

---

## 9. AI-Driven Parameterization for Non-Canonical Amino Acids (NCAAs)

*   **The Feature:** Integrate fast ML surrogates for quantum mechanics (like ANI-2x, MACE, or
    AIMNet2) to automatically calculate partial charges and generate AMBER/CHARMM-compatible
    force-field parameters for novel NCAAs on the fly, without requiring expensive Gaussian or
    ORCA calculations.
*   **Missing Detail — SMILES-to-Parameter Pipeline:** Accept a SMILES string for an arbitrary
    NCAA, run the QM surrogate to compute partial charges and torsional potentials, then directly
    inject the result into OpenMM as a custom `ForceField` XML — fully automated and scriptable.
*   **Missing Detail — D-Amino Acid Extension:** `synth-pdb` already supports D-amino acids.
    Extend the NCAA framework to cover common therapeutic NCAAs: Aib (α-methylalanine),
    β-amino acids, N-methylated residues, and staple-crosslinker residues (pentenylglycine
    for RCM stapling). These are critical for cell-penetrating peptides and stapled peptide
    therapeutics.
*   **Missing Detail — PTM Auto-Parameterization:** The existing `docking.py` already handles
    SEP, TPO, and PTR via residue name mapping. Formalize and extend this into a full PTM
    auto-parameterization framework covering phosphorylation, glycosylation, ubiquitination,
    and acetylation, each generating correct charge distributions and van der Waals parameters.
*   **Why it's competitive:** Makes `synth-pdb` the undisputed tool of choice for synthetic
    biologists and peptide drug designers working beyond the standard 20 amino acids — a major
    weakness of every current monolithic model including AlphaFold 3.

---

## 10. Protein–Protein & Protein–Ligand Interaction Modeling

The existing `docking.py` module currently only handles PQR file generation (charge assignment).

*   **The Feature:** Extend `docking.py` into a full, AI-assisted protein–protein (PPI) and
    protein–small molecule interaction (PLI) modelling suite, using lightweight neural docking
    engines rather than expensive rigid-body search.
*   **Missing Detail — Neural Docking (DiffDock / EquiBind style):** Integrate a pre-trained
    SE(3)-equivariant neural docking model that places a small molecule ligand into a binding
    pocket in milliseconds. The existing PQR/charge infrastructure in `docking.py` makes this
    a natural extension.
*   **Missing Detail — Synthetic PPI Training Data:** Use the existing `generator.py` and
    `msa.py` to generate balanced datasets of `(Interacting Pair, Non-Interacting Pair)` protein
    complexes for training binary PPI classifiers. This is exactly the kind of task the Data
    Factory philosophy is built for.
*   **Missing Detail — Interface SASA & Hot-Spot Prediction:** Add an `interface` analysis module
    that computes buried interface SASA, identifies hot-spot residues (those contributing >2
    kcal/mol to binding energy via alanine scanning), and flags them in the output report. This
    is critical for antibody–antigen and therapeutic peptide design.
*   **Why it's useful:** Expands the audience from protein *structure* researchers to protein
    *interaction* researchers — a significantly larger community — and directly supports drug
    discovery applications.

---

## 11. Federated & Privacy-Preserving Synthetic Data Generation

*   **The Feature:** Provide a `synth-pdb` workflow that generates privacy-preserving synthetic
    structural biology datasets from *private* experimental data (e.g., proprietary NMR spectra
    or unpublished crystal structures) without exposing the raw data. Use differential privacy
    (DP-SGD during surrogate training) and synthetic data auditing to certify that the generated
    dataset cannot be reverse-engineered to reveal the original structure.
*   **Why it's novel:** Pharmaceutical companies routinely cannot share proprietary structural
    data to train collaborative AI models. A privacy-preserving synthetic data pipeline would
    unlock massive industrial adoption and is a genuinely unexplored niche in structural biology.
*   **Implementation Path:** Build on the existing `dataset.py` DatasetGenerator. Add a
    `PrivacySynth` wrapper that trains a small diffusion model on the private dataset using
    DP-SGD (Opacus library), then samples from the trained model to generate a published
    synthetic dataset with formal DP guarantees (ε, δ).

---

## 12. Benchmarking Suite Against AlphaFold / ESMFold

*   **The Feature:** A formal, reproducible benchmarking pipeline that evaluates any structure
    prediction model (AlphaFold2, ESMFold, RoseTTAFold, OmegaFold) against `synth-pdb`'s own
    ground-truth synthetic structures and spectra. Produce standardized CASP-style metrics
    (GDT-TS, TM-score, lDDT) alongside NMR-specific metrics (chemical shift RMSD, RDC Q-factor).
*   **Missing Detail — "Can AlphaFold Predict synth-pdb?" Challenge:** Frame this as a challenge:
    given only the sequence of a synthetically generated `synth-pdb` structure, can AlphaFold
    recover the correct fold? Because `synth-pdb` controls the ground truth, the benchmark is
    perfectly objective. Publish results as a pre-print — this would generate enormous community
    interest.
*   **Missing Detail — Spectroscopic Benchmark:** Extend beyond structure to spectroscopy. Given
    a predicted structure from AlphaFold, compute synthetic NMR chemical shifts and compare to
    the `synth-pdb` ground truth. This creates a novel "NMR Accuracy Benchmark" that tests
    whether structural AI models are biophysically plausible, not just geometrically correct.
*   **Why it's strategic:** This positions `synth-pdb` as the *reference implementation* for
    benchmarking in structural AI — a role that would guarantee citations and ongoing adoption
    regardless of which prediction model is currently dominant.

---

## Strategic Impact Analysis

| Priority | Feature | Reward/Risk | Primary Value |
| :---: | :--- | :--- | :--- |
| ⭐⭐⭐ | **Neural Scoring / GNN (Item 7)** | **Very High** | Infrastructure already built (`quality/gnn`). Pre-trained weights + pLDDT output = immediate user value. |
| ⭐⭐⭐ | **AlphaFold Benchmarking Suite (Item 12)** | **Very High** | Community positioning as the reference benchmark; generates citations and press. Low code risk. |
| ⭐⭐ | **AI Spectroscopy Surrogates (Item 5)** | **High** | Becomes the "Data Foundry" for every multimodal NMR/Cryo-EM AI lab. Foundation in `cryo_em.py`. |
| ⭐⭐ | **IDP Ensembles + SAXS Reweighting (Item 6)** | **High** | Captures the IDP/LLPS pharmaceutical audience. Existing `ensemble` package is a strong foundation. |
| ⭐⭐ | **NCAA Parameterization (Item 9)** | **Medium-High** | Captures synthetic biology/therapeutic peptide market. D-amino support already in place. |
| ⭐ | **Inverse Folding — Multi-State (Item 2)** | **Medium** | Transformative capability but requires significant ML infrastructure. |
| ⭐ | **Differentiable Physics (Item 4)** | **Medium** | Essential for deep learning researchers; high maintenance cost. |
| ⭐ | **Federated Synthetic Data (Item 11)** | **Medium** | Genuinely novel; requires pharmaceutical partnerships to validate. |

---

## 🏆 Highest Reward/Risk Recommendation

**Item 12 (AlphaFold Benchmarking Suite)** paired with **Item 7 (GNN Scorer with Pre-Trained
Weights)** is the highest-leverage combination.

**Why Item 12 wins on strategic impact:**

1.  **Zero competition:** No existing tool publishes a standardized benchmark comparing
    structure prediction AI against a synthetic ground truth with matched NMR observables.
2.  **Guaranteed citations:** Any lab using AlphaFold (essentially everyone) would cite
    a paper establishing this benchmark, cementing `synth-pdb` as infrastructure.
3.  **Low technical risk:** `synth-pdb` already generates the ground-truth structures and
    spectra. The only new work is the evaluation harness and the comparison write-up.
4.  **Community flywheel:** Publishing the benchmark as a pre-print would drive GitHub stars,
    citations, and adoption — which in turn validates every other feature on this roadmap.

**Why Item 7 wins on immediate user value:**

The `quality/gnn` subpackage already exists with `model.py`, `graph.py`, and `gnn_classifier.py`.
The gap between the current state and a highly valuable, polished feature is small:
bundle pre-trained weights → expose a clean `score(pdb_path) → float` API → add per-residue
pLDDT output. This single addition would make `synth-pdb` the only open-source tool that
provides both *generation* and *instant neural quality assessment* in a unified library.

### Summary of Strategic Positioning

By adopting these features, `synth-pdb` avoids competing with AlphaFold directly. Instead, it
becomes the **"Pico-Physics Engine"**, **"Data Foundry"**, and **"Reference Benchmark"** that
every AI researcher uses to train, test, and audit their own structural biology models.
