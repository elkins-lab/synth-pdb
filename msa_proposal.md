# The Most Impressive Feature: Co-Evolutionary Constraints (Synthetic MSA Generation) 🧬

Looking at the intersection of your ROADMAP (specifically *Section 2: AI Research Support*), the trajectory of structural biology, and the capabilities of `synth-pdb`, the single most impressive feature you could implement is **Synthetic Multiple Sequence Alignment (MSA) Generation with Co-Evolutionary Constraints**.

## Why is this so impressive?

The entire revolution of modern structural biology AI (AlphaFold-2, RoseTTAFold, ESMFold) is built on **co-evolution**. When a protein sequence mutates over millions of years, if one amino acid in the core mutates to a larger size, its spatial neighbor must mutate to a smaller size to prevent a steric clash. AI models learn the 3D structure by finding these correlated mutations in an MSA.

Right now, `synth-pdb` gives researchers 3D structures and 1D sequences, but to truly test models like AlphaFold, they need the *evolutionary history*.

### The Feature:

You would implement a feature (`synth-pdb --gen-msa --msa-depth 1000`) that:
1. Generates the base 3D physical structure (which you already do).
2. Calculates the pairwise contact map (which you already do).
3. **Simulates Evolutionary Drift**: It takes the starting sequence and slowly mutates it, generating 1,000 homologous sequences.
4. **The Magic Step**: When it mutates a residue, it looks at the *contact map*. If residue $i$ mutates, the probability of residue $j$ (its spatial neighbor) mutating in a complementary way is drastically increased.
5. **Output**: A `.fasta` file containing a synthetic Multiple Sequence Alignment (MSA) that perfectly encodes the 3D structure of your synthetic protein.

### The Academic Impact

This is a "holy grail" feature for AI mechanism researchers. Currently, tracing exactly *how* a neural network extracts a contact map from an MSA is difficult because natural MSAs are messy, full of phylogeny biases, missing data, and noise.

By providing **perfect, synthetic MSAs** with mathematically controlled co-evolution signals, you provide a clean dataset for researchers to "debug" AI models like AlphaFold. They can ask: "If I inject exactly a 15% co-evolution signal into this synthetic MSA, at what point does AlphaFold successfully predict the fold?"

### Why it fits your architecture perfectly:

You have already built the hardest parts of this:
*   You have the vectorized `BatchedGenerator`.
*   You have the `contact_map` extraction in `dataset.py`.
*   You have the physics engine ensuring the ground-truth structure is valid.

All that is required is writing the Markov Chain logic to mutate the sequence while respecting the contact map constraints!
