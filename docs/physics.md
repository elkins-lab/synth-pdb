# Biophysics 101

## Understanding Energy Minimization

**Energy Minimization** is the process of moving atoms "downhill" to find the nearest stable shape.

```text
      High Energy
      (Unstable)
          |
         / \       Forces push atoms "downhill"
        /   \     (Gradient Descent)
       /     \
      /       \___
     /            \
    /              \__ Low Energy
   /                  (Stable / Minimized)
```

`synth-pdb` defaults to **Implicit Solvent (OBC2)** to simulate the effect of water without the computational cost of thousands of explicit water molecules. This allows for lightning-fast physics refinement natively on the CPU.

## Explicit Solvent (Water Box)

For advanced biophysics and molecular dynamics (MD), `synth-pdb` now fully supports generating and simulating **Explicit Solvent**.

By using `--solvent explicit`, OpenMM will build a surrounding box of classical **TIP3P** water molecules around the generated peptide.

```bash
# Generate a linear peptide padded by 1.2 nanometers of water
synth-pdb --sequence ALA-PRO-GLY --minimize \
  --solvent explicit \
  --solvent-padding 1.2
```

> [!TIP]
> **Stripping Solvent:** 
> By default, `synth-pdb` strips the thousands of generated `HOH` (water) atoms from the final `.pdb` output to keep file sizes small and clean for downstream AI pipelines. If you want to export the entire simulated water box, use the `--keep-solvent` flag.

## The Generation Pipeline

```text
[User] -> [Generator] -> [Geometry Builder] -> [Sidechain Packer] -> [Energy Minimizer] -> [PDB File]
             ^                  |                    |                      |
             |              (N-CA-C-O)           (Rotamers)             (OpenMM)
             |                                       |                      |
             +---------------------------------------+----------------------+
```
