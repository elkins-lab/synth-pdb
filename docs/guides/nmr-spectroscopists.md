# Guide for NMR Spectroscopists

How to use `synth-pdb` to generate structures with realistic synthetic NMR observables
for training datasets, validation pipelines, and spectral simulation.

---

## Quick-Start: Generating a Structure with All NMR Data

```bash
synth-pdb \
  --sequence MGSSHHHHHHSSGLVPRGSH \
  --minimize \
  --gen-nef \
  --gen-relax \
  --gen-shifts --shift-predictor shiftx2 \
  --gen-couplings \
  --output-rdcs rdcs.csv \
  --output peptide.pdb
```

This produces:
- `peptide.pdb` — coordinates
- `peptide_nef.nef` — NOE restraints in NEF format
- `peptide_relax.nef` — R₁, R₂, NOE relaxation data in NEF
- `peptide_shifts.nef` — ¹H, ¹⁵N, ¹³Cα, ¹³Cβ, ¹³C chemical shifts in NEF
- `peptide_couplings.csv` — ³J(HN,HA) couplings for each residue
- `rdcs.csv` — backbone ¹D_NH RDCs for each residue

---

## Controlling Chemical Shift Prediction: `--shift-predictor`

When `--gen-shifts` is active, you choose which predictor backend `synth-nmr` uses.

| Flag | Backend | Best For |
|---|---|---|
| `--shift-predictor shiftx2` | SHIFTX2 (Han et al., 2011) | Best accuracy; needs binary |
| `--shift-predictor empirical` | SPARTA+ / empirical tables | Reproducible CI, no external binary |

**Default** is `shiftx2`, which automatically falls back to the empirical method if
the SHIFTX2 binary is not installed.

```bash
# Use empirical method explicitly (e.g., in Docker CI where SHIFTX2 not installed)
synth-pdb --sequence ACDEFGHIKLM --gen-shifts --shift-predictor empirical \
  --output peptide.pdb
```

**Accuracy comparison** (RMSD from BMRB experimental shifts):

| Nucleus | SHIFTX2 | SPARTA+ |
|---|---|---|
| ¹H | 0.04 ppm | 0.05 ppm |
| ¹³C | 0.44 ppm | 0.55 ppm |
| ¹⁵N | 1.17 ppm | 2.06 ppm |

*Sources: Han et al. (2011), Shen & Bax (2010)*

---

## Generating Synthetic RDC Data: `--output-rdcs`

RDCs are backbone ${}^1D_\text{NH}$ values (Hz), computed from the N–H bond vector
orientation relative to an alignment tensor.

```bash
synth-pdb \
  --sequence ACKNILQ \
  --minimize \
  --output-rdcs rdcs.csv \
  --rdc-da 12.0 \
  --rdc-r 0.15 \
  --output peptide.pdb
```

> **Note:** `--minimize` is recommended with `--output-rdcs` to ensure the structure
> has backbone amide H atoms added by OpenMM.

### Alignment Tensor Parameters

| Flag | Default | Meaning |
|---|---|---|
| `--rdc-da` | `10.0` | Axial component $D_a$ in Hz (typical: 5–25 Hz) |
| `--rdc-r` | `0.1` | Rhombicity $R$, $0 \leq R \leq 2/3$ |

The RDC formula (Tjandra & Bax, 1997, *Science* 278:1111):

$$D(\theta, \phi) = D_a\left[(3\cos^2\theta - 1) + \frac{3}{2}R\sin^2\theta\cos 2\phi\right]$$

### RDC Output CSV Format

```
res_id,residue,RDC_NH_Hz
1,A,18.4231
2,C,7.1203
3,K,-3.8820
4,N,12.4455
...
```

Proline residues are automatically excluded (no backbone amide H).

### Physical Range

For $D_a = 10$ Hz, $R = 0.1$: values span approximately $-11.5$ to $+20$ Hz.
Values outside the range $[D_a(-1-1.5R),\; 2D_a]$ indicate an error.

---

## Interpreting the PDB REMARK Header

Every `synth-pdb` output PDB file embeds a `REMARK 3` block recording the exact
command-line invocation:

```
REMARK 3  GENERATION PARAMETERS:
REMARK 3  Command:
REMARK 3    synth-pdb --sequence ACKNILQ --minimize --output-rdcs rdcs.csv
REMARK 3    --rdc-da 12.0 --rdc-r 0.15 --gen-shifts --output peptide.pdb
```

This satisfies **FAIR data principles** (Findable, Accessible, Interoperable, Reusable):
any researcher can exactly reproduce the generation from the REMARK block alone.

---

## Further Reading

- [NMR Theory: Residual Dipolar Couplings](../science/nmr-theory.md#residual-dipolar-couplings) — physics, derivation, literature
- [API: synth_pdb.rdc](../api/nmr.md#synth_pdbrdc--residual-dipolar-couplings) — module reference
- Han, B. et al. (2011). SHIFTX2. *J Biomol NMR*, 50, 43–57. DOI: [10.1007/s10858-011-9478-4](https://doi.org/10.1007/s10858-011-9478-4)
- Tjandra, N. & Bax, A. (1997). RDC measurement in liquid crystals. *Science*, 278, 1111. DOI: [10.1126/science.278.5340.1111](https://doi.org/10.1126/science.278.5340.1111)
