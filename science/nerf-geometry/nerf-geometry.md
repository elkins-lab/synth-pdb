# NeRF Geometry

The **NeRF (Natural Extension Reference Frame)** algorithm is a fundamental technique for building 3D molecular structures from internal coordinates.

## Overview

NeRF converts internal coordinates (bond lengths, angles, and dihedrals) into 3D Cartesian coordinates. This is essential for protein structure generation because:

- **Compact representation**: Internal coordinates are more compact than Cartesian coordinates
- **Chemical validity**: Bond lengths and angles follow known chemical constraints
- **Efficient sampling**: Sampling in dihedral space is more efficient than Cartesian space

## Mathematical Foundation

### Internal Coordinates

For a chain of atoms, we define:

1. **Bond Length** ($r$): Distance between consecutive atoms
   - Example: N-CA bond = 1.46 Å
   
2. **Bond Angle** ($\theta$): Angle formed by three consecutive atoms
   - Example: N-CA-C angle = 111°
   
3. **Dihedral Angle** ($\phi$): Torsion angle formed by four consecutive atoms
   - Example: Phi (φ) and Psi (ψ) backbone dihedrals

### The NeRF Algorithm

Given three atoms **A**, **B**, **C** with known positions, and internal coordinates for a new atom **D**:

- Bond length: $r_{CD}$
- Bond angle: $\theta_{BCD}$
- Dihedral angle: $\phi_{ABCD}$

The algorithm computes the position of **D** as follows:

1. **Create local coordinate system** at C:
   - **z-axis**: Along the C→B direction
   - **x-axis**: In the plane of A-B-C, perpendicular to z
   - **y-axis**: Perpendicular to both x and z

2. **Place D in local coordinates**:
   ```
   D_local = [
       r * sin(θ) * cos(φ),
       r * sin(θ) * sin(φ),
       r * cos(θ)
   ]
   ```

3. **Transform to global coordinates**:
   - Apply rotation matrix to align local axes with global axes
   - Translate by position of C

## Application to Proteins

### Backbone Construction

The protein backbone is built iteratively:

```
N₁ → CA₁ → C₁ → N₂ → CA₂ → C₂ → ...
```

Each atom is placed using NeRF with:
- **Fixed bond lengths**: From crystallographic data
- **Fixed bond angles**: From chemical constraints
- **Variable dihedrals**: Sampled from Ramachandran distributions

### Side-Chain Placement

Side-chains are added using:
- **Rotamer libraries**: Pre-computed favorable conformations
- **NeRF algorithm**: To place each side-chain atom
- **Steric constraints**: To avoid clashes

## Implementation in synth-pdb

The `geometry` module implements NeRF for protein construction:

```python
from synth_pdb.geometry import place_atom

# Place a new atom given three reference atoms
new_position = place_atom(
    atom_a=pos_a,  # Position of atom A
    atom_b=pos_b,  # Position of atom B
    atom_c=pos_c,  # Position of atom C
    bond_length=1.52,  # C-C bond
    bond_angle=111.0,  # degrees
    dihedral=180.0     # degrees
)
```

## Advantages

1. **Chemical validity**: Structures automatically satisfy bond constraints
2. **Efficiency**: O(n) complexity for n atoms
3. **Interpretability**: Dihedrals directly correspond to conformational freedom
4. **Sampling**: Easy to sample conformational space

## Limitations

1. **Rigid geometry**: Bond lengths and angles are typically fixed
2. **Sequential construction**: Requires a defined atom ordering
3. **Numerical precision**: Small errors can accumulate in long chains

## See Also

- [Ramachandran Plots](ramachandran.md) - Dihedral angle distributions
- [Rotamer Libraries](rotamers.md) - Side-chain conformations
- [geometry Module](../api/geometry.md) - Implementation details

## Advanced Topics

### Z-Matrix Construction
Proteins are defined by their "Internal Coordinates" (Z-Matrix):
1. **Bond Length**: distance between two atoms.
2. **Bond Angle**: angle between three atoms.
3. **Torsion/Dihedral Angle**: twist between four atoms.

Our generator builds structures by transitioning from this 1D/2D internal representation into 3D Cartesian space. This algorithm is the engine of our protein builder, allowing us to "walk down" the chain atom-by-atom with mathematical precision.

### Circular Statistics (The 180/-180 Problem)
In protein geometry, torsion angles (Phi, Psi, Omega, Chi) are periodic. This introduces a challenge for both math and AI modeling:

1. **The Boundary Artifact**: An angle of -179 deg is physically very close to +179 deg, but their arithmetic difference is 358 deg.
2. **Correct Distance**: To find the "real" difference between two angles, we must use: `diff = (a - b + 180) % 360 - 180`.
3. **AI Loss Functions**: Naive Mean Squared Error (MSE) fails on angles because it doesn't understand this wrapping. High-performance models (like AlphaFold) often predict the (Sine, Cosine) of the angle instead, ensuring a smooth, continuous coordinate space.
4. **Phase Wrapping**: In structure generation, "Drift" must be applied carefully to avoid discontinuities at the -180/180 boundary.

### SIMD & Parallel Geometry
Traditional biology code uses "Serial Geometry" ($O(B \times L)$). To place atoms for $B$ structures of length $L$, it loops $B$ times.

`synth-pdb` uses **Single Instruction, Multiple Data (SIMD)** logic:
1. **Broad Geometry**: We treat the coordinates as a massive block of numbers rather than individual XYZ points.
2. **Vector Units**: Hardware like the M4's AMX or a GPU's CUDA cores can execute one operation (e.g., a cross product) across thousands of data points at once.
3. **Efficiency**: By avoiding the Python interpreter loop for each structure, we reach throughput levels required for "Foundation Model" training in proteomics.

On modern hardware (Apple M4 AMX, NVIDIA Tensor Cores), serial loops are extremely inefficient. By vectorizing the math into large matrix operations, memory bandwidth is maximized via contiguous array access and hardware acceleration (Accelerate/MPS/Metal) can be leveraged automatically.

## References

1. Parsons, J., et al. (2005). "Practical conversion from torsion space to Cartesian space for in silico protein synthesis." *Journal of Computational Chemistry*, 26(10), 1063-1068.
2. Coutsias, E. A., et al. (2004). "Using quaternions to calculate RMSD." *Journal of Computational Chemistry*, 25(15), 1849-1857.
3. Kabsch, W. (1976). "A solution for the best rotation to relate two sets of vectors." *Acta Crystallographica Section A*, 32(6), 922-923.
4. Kabsch, W. (1978). "A discussion of the solution for the best rotation to relate two sets of vectors." *Acta Crystallographica Section A*, 34(5), 827-828.

