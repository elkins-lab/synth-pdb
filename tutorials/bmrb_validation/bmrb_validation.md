# BMRB Validation Pipeline

In this tutorial, we demonstrate how to validate synthetic protein models by programmatically fetching experimental data from the **Biological Magnetic Resonance Data Bank (BMRB)**.

## Prerequisites

You will need the following packages:
```bash
pip install synth-pdb pynmrstar requests
```

## 1. Fetching BMRB Entry Metadata

The `BMRBAPI` module allows us to fetch peer-reviewed assignments for structural validation. Let's start with human **Ubiquitin** (BMRB Entry 6457).

```python
from synth_pdb.bmrb_api import BMRBAPI

# Fetch entry metadata
metadata = BMRBAPI.get_entry_metadata("6457")
print(f"Title: {metadata.get('title')}")
print(f"Authors: {', '.join([a['first_name'] + ' ' + a['last_name'] for a in metadata.get('authors', [])])}")
```

## 2. Downloading Experimental Chemical Shifts

Chemical shifts are the "fingerprints" of a protein's structure. We can fetch the assigned shifts and compare them to our synthetic predictions.

```python
# Fetch experimental shifts
shifts = BMRBAPI.fetch_chemical_shifts("6457")

# Check shifts for the first residue (Met 1)
if 1 in shifts:
    print(f"Experimental Shifts for Met 1: {shifts[1]}")
```

## 3. Retrieving Distance Restraints (NOEs)

Distance restraints define the 3D fold. We can fetch the ground-truth upper bounds used by the original researchers.

```python
# Fetch experimental distance restraints
restraints = BMRBAPI.fetch_restraints("6457")
print(f"Retrieved {len(restraints)} experimental NOE restraints.")

# Example: Look at the first restraint
if restraints:
    r = restraints[0]
    print(f"Restraint between {r['atom_name_1']}{r['index_1']} and {r['atom_name_2']}{r['index_2']}: Max {r['upper_limit']} Å")
```

## 4. Comparing Synthetic vs. Experimental Data

Now we generate a synthetic Ubiquitin structure and check if it satisfies the experimental NOE restraints.

```python
from synth_pdb.generator import PeptideGenerator
from synth_pdb.analysis import GeometryAnalyzer

# 1. Generate synthetic structure (Human Ubiquitin sequence)
ubiq_seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
gen = PeptideGenerator(ubiq_seq)
structure = gen.generate(conformation="alpha") # Start with a simple alpha helix

# 2. Check a restraint (e.g., a known long-range contact)
# (In a real pipeline, we would loop over all fetched restraints)
target_restraint = [r for r in restraints if abs(r['index_1'] - r['index_2']) > 10][0]
print(f"\nChecking restraint: {target_restraint}")
```

## 5. Benchmarking Geometric Quality

Finally, we can use the `PDBValidationAPI` to see how well-known experimental structures (like 1D3Z) rank in terms of geometric quality.

```python
from synth_pdb.bmrb_api import PDBValidationAPI

# Fetch PDBe validation summary for 1D3Z
validation = PDBValidationAPI.get_validation_summary("1D3Z")
if validation:
    # Percentiles tell us how the structure ranks globally (higher is better)
    clash_percentile = validation[0].get("absolute_percentile_clashscore")
    rama_percentile = validation[0].get("absolute_percentile_ramachandran")
    
    print(f"1D3Z Global Clashscore Percentile: {clash_percentile}")
    print(f"1D3Z Global Ramachandran Percentile: {rama_percentile}")
```

## Summary

By combining `bmrb_api` with `synth-pdb`, you can:
-   **Benchmark** your prediction algorithms against peer-reviewed data.
-   **Validate** that synthetic ensembles are biologically plausible.
-   **Test** structure calculation protocols with "perfectly assigned" experimental data.
