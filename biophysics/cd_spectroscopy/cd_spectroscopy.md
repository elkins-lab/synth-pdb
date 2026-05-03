# Circular Dichroism (CD) Spectroscopy

Circular Dichroism (CD) is a spectroscopic technique used to investigate the secondary structure, folding, and binding properties of proteins. `synth-pdb` includes a synthetic CD simulator that allows researchers and students to visualize the expected far-UV CD spectrum of a protein model.

## Scientific Basis

Protein Circular Dichroism in the far-UV region (190–250 nm) is dominated by the absorption of the peptide bond. The characteristic CD signatures of secondary structure elements arise from the excitonic coupling of the amide chromophores in a regular geometric arrangement.

### Basis Spectra

The `synth-pdb` simulator uses the "basis spectra" method to synthesize the overall spectrum. This approach assumes that the total molar ellipticity $[\theta]$ of a protein is a linear combination of the ellipticities of its constituent secondary structures:

$$ [\theta]_{total}(\lambda) = f_{helix} \cdot [\theta]_{helix}(\lambda) + f_{sheet} \cdot [\theta_{sheet}](\lambda) + f_{coil} \cdot [\theta_{coil}](\lambda) $$

Where $f_i$ is the fraction of each secondary structure element.

The basis spectra used in this module are derived from the foundational work of **Greenfield and Fasman (1969)**.

| Element | Wavelength Maxima/Minima | Characteristic Signature |
| :--- | :--- | :--- |
| **$\alpha$-Helix** | 222 nm (-), 208 nm (-), 192 nm (+) | Two negative peaks and a strong positive peak |
| **$\beta$-Sheet** | 217 nm (-), 195 nm (+) | A single negative peak and a positive peak |
| **Random Coil** | 198 nm (-), ~220 nm (weak +) | Strong negative peak at low wavelengths |

## Generating Synthetic CD Results

To generate a synthetic CD spectrum and plot, use the `--gen-cd` flag.

```bash
# Generate a Poly-Alanine helix and its CD spectrum
synth-pdb --sequence "AAAAAAAAAAAAAAAAAAAA" --conformation alpha --gen-cd
```

This will produce:
1. `generated_structure.pdb`
2. `generated_structure_cd.png` (A plot of the spectrum)

### Analysis Output

When `--gen-cd` is active, `synth-pdb` also prints a **Scientific Validation Report** to the terminal. This report compares the synthetic minima to literature benchmarks (e.g., verifying if a pure helix matches the expected -36,000 $deg \cdot cm^2 / dmol$ at 222 nm).

## References

1.  **Greenfield, N., & Fasman, G. D. (1969).** Computed circular dichroism spectra for the evaluation of protein conformation. *Biochemistry*, 8(10), 4108-4116.
2.  **Provencher, S. W., & Glöckner, J. (1981).** Estimation of globular protein secondary structure from circular dichroism. *Biochemistry*, 20(1), 33-37.
3.  **Kelly, S. M., Jess, T. J., & Price, N. C. (2005).** How to study proteins by circular dichroism. *Biochimica et Biophysica Acta (BBA)-Proteins and Proteomics*, 1751(2), 119-139.
