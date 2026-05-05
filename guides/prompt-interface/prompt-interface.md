# Prompt-to-Protein Interface

The `synth-pdb` tool features a powerful, natural language interface that allows you to generate complex protein structures and simulated data by simply describing what you want in plain English. 

Instead of memorizing dozens of complex command-line flags, you can pass a `--prompt` argument and the system will automatically configure the generation engine for you.

## Quick Start

### 1. Choose a Backend

#### Option A: Cloud API (OpenAI)
By default, if you have an OpenAI API key, you can use the ultra-fast cloud backend.

1. Set your API key in your environment:
```bash
export OPENAI_API_KEY="sk-your-api-key"
```

#### Option B: Local SLM (Offline & Private)
For total privacy and zero API costs, `synth-pdb` supports downloading and running a highly compressed **Small Language Model (SLM)** locally on your CPU or GPU.

1. Install the optional local AI dependencies:
```bash
pip install synth-pdb[local-llm]
```

2. Use the `--llm-backend local` flag in your commands.

*Note: The first time you use the local backend, it will automatically download a ~2.4GB GGUF model file (like Phi-3-Mini) to your HuggingFace cache. Subsequent runs will load it instantly.*

### 2. Provide your Prompt

There are three ways to pass a natural language request to `synth-pdb`.

#### A. Inline Argument (Standard)
Pass the prompt directly on the command line:
```bash
synth-pdb --prompt "Build a 30-residue cyclic peptide in a beta-sheet conformation."
```

#### B. Piped Input
Pipe the output of another tool or a file into `synth-pdb`:
```bash
echo "Generate a 15-mer alpha helix" | synth-pdb --prompt
```

#### C. Interactive Mode
Run `--prompt` without a value to open an interactive buffer. This is useful for complex, multi-line instructions:
```bash
synth-pdb --prompt
# Enter your natural language prompt (type 'exit' on a new line or press Ctrl+D to finish):
# > Build an alpha helix with 
# > a zinc binding motif.
# > exit
```

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

## Tips for Reliable Generation

While the Prompt-to-Protein interface is designed to be intuitive, performance varies depending on the LLM backend you choose. 

### 1. Verify the Interpretation
When you run a prompt, `synth-pdb` will log exactly how it interpreted your request. **Always check this line** to ensure the model understood your structural requirements:

```bash
Translated prompt into: --length 20 --conformation alpha
```

### 2. Handling "SLM Hallucinations" (Local Backend)
The local backend uses Small Language Models (SLMs) like `Phi-3`. These models are efficient but can occasionally hallucinate long sequences or "forget" to set structural flags if the prompt is too conversational.

*   **Avoid Conversation Filler:** Instead of "Hey there, would you please generate me a...", try clinical phrasing: **"Length 30, conformation beta, minimize"**.
*   **Sequence Over-Generation:** If the model starts generating an extremely long sequence (e.g., "Successfully predicted S2 for 225 residues") when you didn't ask for one, try being more explicit about the length: **"Generate exactly 20 residues"**.
*   **Defaulting to Alpha:** If you don't see `--conformation beta` (or your requested type) in the translation logs, the system will default to an **Alpha Helix**.

### 3. Use the Cloud for Precision
If you need high-precision mapping of multiple complex requirements (e.g., specific solvent models, rdc tensors, and PTM rates all in one prompt), the `openai` backend (using GPT-4o-mini) is significantly more robust and less prone to hallucinations than the smaller local models.

### 4. Nonsense Prompts
If you provide a prompt that contains no biological or structural instructions (e.g., "The quick brown fox..."), the system will log a warning:

```bash
WARNING: No specific structural instructions identified in prompt. Using defaults.
```
In this case, it will fall back to a standard 10-residue linear peptide.

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
