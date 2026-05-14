"""
Dihedral Angle Order Parameter (DAOP) analysis.

Implements the circular-statistics order parameter that quantifies how
consistently a dihedral angle is sampled across an NMR ensemble.  A value
near 1.0 means all models agree; near 0.0 means completely disordered.

Reference:
    Hyberts, S.G., Goldberg, M.S., Havel, T.F. & Wagner, G. (1992).
    The solution structure of eglin c based on measurements of many NOEs
    and coupling constants and its comparison with X-ray structures.
    *Protein Science*, **1**, 736-751.
    https://doi.org/10.1002/pro.5560010606

Note:
    All functions operate on raw angle arrays in **radians**.  They are
    completely independent of file formats, PDB parsers, or any other
    domain infrastructure - they are pure numpy.
"""

import numpy as np
import numpy.typing as npt


class DAOPCalculator:
    """
    Calculator for Dihedral Angle Order Parameters (DAOP).

    The DAOP quantifies the consistency of a dihedral angle across an NMR
    ensemble.  Values range from 0 (completely disordered) to 1 (perfectly
    ordered).  Typical NMR convention: S >= 0.9 means the residue is
    well-defined (corresponds to <~ +/-24deg circular standard deviation).

    All methods are static - no instantiation required.

    Examples:
        >>> import numpy as np
        >>> from synth_pdb.ensemble.daop import DAOPCalculator

        >>> # Perfectly ordered angles (all -60deg)
        >>> angles = np.full(20, -np.pi / 3)
        >>> DAOPCalculator.calculate_order_parameter(angles)
        1.0

        >>> # Find which residues are well-defined across an ensemble
        >>> phi = np.random.normal(-np.pi / 3, np.pi / 36, (50, 20))
        >>> psi = np.random.normal( np.pi / 3, np.pi / 36, (50, 20))
        >>> mask = DAOPCalculator.find_well_defined_residues(phi, psi)
        >>> mask.all()
        True
    """

    @staticmethod
    def calculate_order_parameter(angles: npt.NDArray[np.float64]) -> float:
        """
        Calculate the circular order parameter S for a set of dihedral angles.

        The formula is the length of the mean resultant vector on the unit circle:

            S(phi_i) = (1/N) * sqrt( (Sum sin phi_i_j)^2 + (Sum cos phi_i_j)^2 )

        where N is the number of models and j iterates over models.  This is
        equivalent to the magnitude of the mean of the complex exponentials
        e^(i*phi), a standard circular-statistics estimator.

        Args:
            angles: 1-D array of dihedral angles **in radians** sampled across
                the ensemble (one value per model for a single residue).

        Returns:
            Order parameter S in [0, 1]:

            * S = 1.0  - perfectly ordered (all angles identical)
            * S >= 0.9  - well-ordered (<~ +/-24deg circular std dev)
            * S ~ 0.0  - completely disordered (random distribution)

        Examples:
            >>> import numpy as np
            >>> from synth_pdb.ensemble.daop import DAOPCalculator

            >>> # All angles identical -> S = 1
            >>> angles = np.full(10, -np.pi / 3)
            >>> DAOPCalculator.calculate_order_parameter(angles)
            1.0

            >>> # Random angles -> S ~ 0
            >>> rng = np.random.default_rng(seed=0)
            >>> random_angles = rng.uniform(0, 2 * np.pi, 10_000)
            >>> S = DAOPCalculator.calculate_order_parameter(random_angles)
            >>> S < 0.05
            True
        """
        if len(angles) == 0:
            return 0.0

        n = len(angles)
        sin_sum = float(np.sum(np.sin(angles)))
        cos_sum = float(np.sum(np.cos(angles)))
        return (1.0 / n) * float(np.sqrt(sin_sum**2 + cos_sum**2))

    @staticmethod
    def find_well_defined_residues(
        phi_angles: npt.NDArray[np.float64],
        psi_angles: npt.NDArray[np.float64],
        threshold: float = 1.8,
    ) -> npt.NDArray[np.bool_]:
        """
        Identify well-defined residues using the PDBStat S(phi) + S(psi) criterion.

        A residue is considered well-defined when:

            S(phi_i) + S(psi_i) >= threshold

        The default threshold of 1.8 follows the PDBStat convention and
        corresponds to both angles having S >= 0.9, i.e. a circular standard
        deviation of <~ +/-24deg for each.

        Args:
            phi_angles: Array of shape (n_residues, n_models) with phi angles
                **in radians**.
            psi_angles: Array of shape (n_residues, n_models) with psi angles
                **in radians**.
            threshold: Sum-of-order-parameters cutoff. Default 1.8 (PDBStat
                convention).

        Returns:
            Boolean array of shape (n_residues,).  ``True`` entries are
            well-defined residues.

        Examples:
            >>> import numpy as np
            >>> from synth_pdb.ensemble.daop import DAOPCalculator

            >>> # Well-ordered ensemble
            >>> rng = np.random.default_rng(42)
            >>> phi = rng.normal(-np.pi / 3, np.pi / 36, (10, 20))
            >>> psi = rng.normal( np.pi / 3, np.pi / 36, (10, 20))
            >>> mask = DAOPCalculator.find_well_defined_residues(phi, psi)
            >>> mask.all()
            True
        """
        n_residues = phi_angles.shape[0]

        s_phi = np.array(
            [DAOPCalculator.calculate_order_parameter(phi_angles[i]) for i in range(n_residues)]
        )
        s_psi = np.array(
            [DAOPCalculator.calculate_order_parameter(psi_angles[i]) for i in range(n_residues)]
        )

        well_defined: npt.NDArray[np.bool_] = (s_phi + s_psi) >= threshold
        return well_defined

    @staticmethod
    def calculate_backbone_daop(
        phi_angles: npt.NDArray[np.float64],
        psi_angles: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Calculate backbone DAOP values for all residues in a structure.

        Convenience wrapper that applies :meth:`calculate_order_parameter`
        independently to each row of the phi and psi angle matrices.

        Args:
            phi_angles: Array of shape (n_residues, n_models) with phi angles
                **in radians**.
            psi_angles: Array of shape (n_residues, n_models) with psi angles
                **in radians**.

        Returns:
            Tuple of ``(S_phi, S_psi)``, each a 1-D array of shape
            (n_residues,) with order-parameter values in [0, 1].

        Examples:
            >>> import numpy as np
            >>> from synth_pdb.ensemble.daop import DAOPCalculator

            >>> phi = np.random.normal(-np.pi / 3, 0.1, (5, 10))
            >>> psi = np.random.normal( np.pi / 3, 0.1, (5, 10))
            >>> s_phi, s_psi = DAOPCalculator.calculate_backbone_daop(phi, psi)
            >>> s_phi.shape
            (5,)
            >>> s_psi.shape
            (5,)
        """
        n_residues = phi_angles.shape[0]

        s_phi = np.array(
            [DAOPCalculator.calculate_order_parameter(phi_angles[i]) for i in range(n_residues)]
        )
        s_psi = np.array(
            [DAOPCalculator.calculate_order_parameter(psi_angles[i]) for i in range(n_residues)]
        )

        return s_phi, s_psi
