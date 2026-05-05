# Prompt-to-Protein Interface

The `synth-pdb` tool features a powerful, natural language interface that allows you to generate complex protein structures and simulated data by simply describing what you want in plain English. 

Instead of memorizing dozens of complex command-line flags, you can pass a `--prompt` argument and the system will automatically configure the generation engine for you.

## Quick Start

### Option A: Cloud API (OpenAI)
By default, if you have an OpenAI API key, you can use the ultra-fast cloud backend.

1. Set your API key in your environment:
```bash
export OPENAI_API_KEY="sk-your-api-key"
```

2. Run your natural language prompt:
```bash
synth-pdb --prompt "Build a 30-residue cyclic peptide in a beta-sheet conformation, minimize it, and generate the RDC coupling data."
```

### Option B: Local SLM (Offline & Private)
For total privacy and zero API costs, `synth-pdb` supports downloading and running a highly compressed **Small Language Model (SLM)** locally on your CPU or GPU.

1. Install the optional local AI dependencies:
```bash
pip install synth-pdb[local-llm]
```

2. Run your prompt using the local backend:
```bash
synth-pdb --prompt "Build an alpha helix" --llm-backend local
```

*Note: The first time you use the local backend, it will automatically download a ~2.4GB GGUF model file (like Phi-3-Mini) to your HuggingFace cache. Subsequent runs will load it instantly.*

## Example Prompts

The natural language interface is designed to understand complex, multi-step biological requirements. Here are some examples of what you can ask it to do:

### 1. Specific Sequences and Secondary Structure
> "Generate the sequence ALAGLYVAL in a beta sheet conformation."
*(Automatically parses `--sequence ALAGLYVAL --conformation beta`)*

### 2. Post-Translational Modifications & Capping
> "Build a 50-residue random coil, but make sure to cap the N and C termini and minimize it so there are no clashes."
*(Automatically parses `--length 50 --conformation random --cap-termini --minimize`)*

### 3. Advanced Biophysics (Cyclization & Metals)
> "Create a 15-mer alpha helix, fold it into a cyclic peptide, and automatically insert a zinc metal ion if there's a binding motif."
*(Automatically parses `--length 15 --conformation alpha --cyclic --metal-ions auto`)*

### 4. Synthetic Data Generation (NMR / CD)
> "I need a 40-residue polyproline II extended structure. Please simulate its Circular Dichroism spectrum and generate its expected NMR chemical shifts."
*(Automatically parses `--length 40 --conformation ppii --gen-cd --gen-shifts`)*

## How it Works (Under the Hood)
The LLM interface is strictly isolated from the core biophysics engine. It uses a **Zero-Shot JSON Translation** technique:

1. The text of your `--prompt` is combined with a strict system prompt containing the `synth-pdb` CLI schema.
2. The LLM translates your English sentence into a valid JSON object.
3. The interface intercepts this JSON, safely parses it, and dynamically injects those keys directly into the execution flow.
4. The generation engine runs normally, unaware that an LLM was used.

## Educational Example: Fine-Tuning Your Own Model
If you wish to train your *own* specialized model (so it deeply understands `synth-pdb` and runs even faster), we have provided a complete educational script to generate the necessary training corpus.

You can find the SLM Fine-Tuning Pipeline script in:
`examples/ml_integration/slm_finetuning_pipeline.py`

This script demonstrates how to:
- Procedurally generate `(Prompt -> JSON)` pairs.
- Format the data into the strict JSONL schema required for HuggingFace/LoRA training.
- Learn the concepts of **GGUF Quantization** and **Low-Rank Adaptation (LoRA)** through inline pedagogical documentation.
