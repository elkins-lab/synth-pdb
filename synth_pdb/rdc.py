"""Residual Dipolar Couplings (RDCs) for synth-pdb.

This module provides a compatibility shim that re-exports the RDC calculation
engine from the synth-nmr package, following the same pattern as
synth_pdb.chemical_shifts and synth_pdb.coupling.

For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr.rdc import calculate_rdcs

See: https://github.com/elkins/synth-nmr

EDUCATIONAL NOTE — What are Residual Dipolar Couplings (RDCs)?
================================================================
In solution NMR, molecules tumble rapidly in all orientations. This isotropic
tumbling averages the large through-space magnetic dipole–dipole interactions
between nuclear spin pairs (e.g., backbone N–H) to exactly zero — which is why
solution NMR spectra are so sharp compared to solid-state.

However, if molecules are placed in an anisotropic medium — such as a dilute
suspension of rod-like liquid crystals (Tjandra & Bax, 1997), filamentous
phage particles (Hansen et al., 2000), or a strained polyacrylamide gel
(Tycko et al., 2000) — they develop a slight statistical preference for a
particular orientation. This partial alignment is described by the
"alignment tensor" A, a symmetric 3×3 matrix.

Because the tumbling is no longer fully isotropic, the dipolar coupling no
longer averages to zero: a small "residual" coupling remains. For a backbone
N–H bond vector, the RDC value is:

    D(θ, φ) = Da · [(3·cos²θ − 1) + (3/2)·R·sin²θ·cos(2φ)]

where:
  θ   = polar angle of the N–H unit vector with respect to the principal
        (Z) axis of the alignment tensor
  φ   = azimuthal angle of the N–H unit vector in the XY plane of the tensor
  Da  = axial component of the alignment tensor in Hz
        (typical values for proteins: 5–25 Hz; Tjandra & Bax, 1997)
  R   = rhombicity of the tensor, 0 ≤ R ≤ 2/3; R=0 gives axially symmetric
        alignment; R=2/3 is the maximum rhombicity

WHY ARE RDCs USEFUL FOR STRUCTURE DETERMINATION?
RDCs encode the orientation of individual bond vectors relative to a shared
global frame — the alignment tensor. This is fundamentally different from NOE
distance restraints, which are purely local. Because of this:
  • NOEs constrain local folding (secondary structure)
  • RDCs constrain global fold topology (tertiary structure)

The two observables are thus complementary: combining them dramatically
improves the accuracy of NMR-derived protein structures. A landmark study
(Bewley & Clore, 2000) demonstrated that RDC-constrained calculations refined
a 100-ns MD ensemble of HIV-1 protease to within 0.4 Å RMSD of the crystal
structure, without any additional NOE data.

ALIGNMENT TENSOR PARAMETERS:
The alignment tensor is parameterised with just five numbers:
  Da (axial component), R (rhombicity), and three Euler angles orienting
  the tensor in the molecular frame. synth-pdb exposes Da and R via
  --rdc-da and --rdc-r CLI flags; the orientation is implicit (the PAS
  of the tensor is assumed to coincide with the coordinate frame of the
  PDB structure).

References:
  1. Tjandra, N. & Bax, A. (1997). Direct measurement of distances and
     angles in biomolecules by NMR in a dilute liquid crystalline medium.
     Science, 278, 1111–1114. DOI: 10.1126/science.278.5340.1111

  2. Prestegard, J.H., Al-Hashimi, H.M. & Tolman, J.R. (2000).
     NMR structures of biomolecules using field oriented media and residual
     dipolar couplings. Q Rev Biophys, 33, 371–424.
     DOI: 10.1017/S0033583500003656

  3. Hansen, M.R. et al. (2000). Filamentous bacteriophage as vehicles for
     RDC-based structure refinement. J Biomol NMR, 14, 85.
     DOI: 10.1023/A:1008313520205

  4. Bewley, C.A. & Clore, G.M. (2000). Determination of the relative
     orientation of the two halves of the domain-swapped dimer of HIV-1
     protease using RDCs. J Am Chem Soc, 122, 6009.
     DOI: 10.1021/ja000635g

  5. Bax, A. & Grishaev, A. (2005). Weak alignment NMR: a hawk-eyed view
     of biomolecular structure. Curr Opin Struct Biol, 15, 563–570.
     DOI: 10.1016/j.sbi.2005.08.006

"""

# Re-export from synth-nmr for backward compatibility.
# This shim preserves a stable synth_pdb.rdc public API even if the
# underlying synth-nmr implementation evolves.
from synth_nmr.rdc import calculate_rdcs

__all__ = [
    "calculate_rdcs",
]
