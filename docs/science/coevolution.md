# Co-Evolution & Synthetic MSAs

The `synth_pdb` Multiple Sequence Alignment (MSA) generator provides a powerful engine for understanding the mapping between protein sequences and 3D structures. By generating massive alignments constrained by artificial evolutionary pressures, we can computationally reconstruct the rules of Direct Coupling Analysis (DCA) and reverse-engineer structural dependencies.

## The Potts Energy Model

At the core of the MSA generator is a statistical **Potts Model**. Each proposed sequence $S$ is evaluated against an energetic landscape defined by its 3D native contact map:

$$E(S) = \sum_i h_i(S_i) + \sum_{i<j} J_{ij}(S_i, S_j)$$

### Solvent Accessible Surface Area (SASA) Pressure: Local Fields ($h_i$)
When building the local preference fields ($h_i$), the engine strictly enforces the hydrophobic core. If a residue's relative SASA is below a defined threshold (highly buried), the model applies massive positive energy penalties to any polar or charged residue ($R, K, D, E, Q, N, H$). 

Across millions of generations of evolutionary drift, this mathematically guarantees that the core sequence profile will converge almost entirely onto hydrophobic residues ($V, I, L, F, M, W, C, A$).

### Pairwise Interactions: Couplings ($J_{ij}$)
If two residues, $i$ and $j$, are in spatial contact within the original 3D fold, they must physically co-evolve to avoid energetic penalties. The engine applies two primary constraints:

1. **Steric Clashes ($E \gg 0$)**: Two massive residues ($W$ and $Y$) packed together will trigger a severe steric penalty. To lower the structural energy back to equilibrium, one residue must undergo a compensatory mutation to a tiny residue ($G$ or $A$).
2. **Electrostatics**:
   - **Salt Bridges ($E < 0$)**: An opposite-charge pairing (e.g., $K$ and $D$) receives a strong energetic *reward*, deepening the co-evolutionary well.
   - **Repulsion ($E > 0$)**: Like-charges packed in contact receive an energetic *penalty*.

## Metropolis-Hastings Markov Chain Monte Carlo

Generating the MSA requires simulated evolutionary drift. We employ a highly-optimized MCMC sampler traversing the sequence space. 

Instead of evaluating the massive $O(N^2)$ energy equation for the entire sequence at every mutational step, the simulation calculates the continuous $O(1)$ $\Delta E$ difference between states. This accelerates MSA generation by over 500x.

### The "Magic Step" (Coupled Mutations)

Navigating deep "valleys" in the sequence space often presents a statistical barrier. For example, drifting from a `[Large, Tiny]` steric pairing to a `[Tiny, Large]` pairing via point mutations requires crossing an astronomically penalized intermediate state (`[Large, Large]`). 

To solve this, the sampler employs **Coupled Mutations**. 

With a tunable probability (default 20%), the MCMC engine skips the single point-mutation phase and instead proposes mutating *both* halves of a contacting pair simultaneously. If the total $\Delta E$ of the double mutation is favorable (or statistically accepted by thermal noise), the simulation leaps across the evolutionary barrier, flawlessly simulating deep and complex co-evolutionary covariance.
