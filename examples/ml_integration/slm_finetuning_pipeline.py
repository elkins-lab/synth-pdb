#!/usr/bin/env python3
"""
Educational Example: Synthesizing a Training Corpus for Small Language Model (SLM) Fine-Tuning.

This script demonstrates how to procedurally generate a high-quality dataset
required to fine-tune a local "Expert Model" for synth-pdb.

### EDUCATIONAL NOTE: LoRA and SLM Fine-Tuning
Why fine-tune a model when you can just prompt GPT-4?
1. Privacy and Cost: A local model (like Phi-3-mini or Gemma 2B) runs entirely offline on
   your own hardware, costing nothing per API call.
2. Speed and Latency: A specialized model ignores chit-chat and directly outputs JSON.
3. LoRA (Low-Rank Adaptation): We don't train the billions of parameters from scratch.
   Instead, we freeze the base model and train tiny "adapter" matrices that learn the
   specific grammar, parameters, and rules of `synth-pdb`. This takes hours instead of months.
4. GGUF Quantization: After training, the model weights are compressed from 16-bit floats
   down to 4-bit integers. A 3.8 Billion parameter model shrinks from ~8GB to ~2GB, fitting
   easily in laptop RAM.

Usage:
    python slm_finetuning_pipeline.py --samples 500 --output train.jsonl
"""

import argparse
import json
import random

# A subset of valid configurations for synth-pdb to train the model on
CONFORMATIONS = ["alpha", "beta", "ppii", "random"]
METAL_IONS = ["auto", "none"]

# Templates for natural language generation. In a real scenario, you'd use
# a larger model (GPT-4) to generate 100x more linguistic diversity.
TEMPLATES = [
    "Make a {length}-mer {conformation} structure.",
    "Generate a {conformation} peptide of length {length}.",
    "I need {length} residues folded into {conformation}, and {cyclic_text}.",
    "Create a {conformation} structure, {length} AA long. {cap_text}",
    "Simulate a {length} amino acid {conformation} sequence. {minimize_text}",
]


def generate_random_example() -> tuple[str, dict]:
    """Generate a pair of (Natural Language Request, synth-pdb JSON)."""
    length = random.randint(5, 50)
    conformation = random.choice(CONFORMATIONS)
    is_cyclic = random.choice([True, False])
    cap_termini = random.choice([True, False])
    minimize = random.choice([True, False])

    # Construct target JSON
    target_json = {
        "length": length,
        "conformation": conformation,
    }

    cyclic_text = "make it cyclic" if is_cyclic else "don't make it cyclic"
    if is_cyclic:
        target_json["cyclic"] = True

    cap_text = "Cap the termini." if cap_termini else "Leave termini uncapped."
    if cap_termini:
        target_json["cap_termini"] = True

    minimize_text = "Run minimization." if minimize else "Skip minimization."
    if minimize:
        target_json["minimize"] = True

    template = random.choice(TEMPLATES)
    prompt = template.format(
        length=length,
        conformation=conformation,
        cyclic_text=cyclic_text,
        cap_text=cap_text,
        minimize_text=minimize_text,
    )

    return prompt, target_json


def build_huggingface_dataset(num_samples: int, output_file: str) -> None:
    """
    Builds the dataset in the standard HuggingFace JSONL format.
    Each line is a conversation with a system prompt, user request, and assistant JSON reply.
    """
    system_prompt = (
        "You are a CLI argument translator for synth-pdb. "
        "Translate the user's natural language request into a valid JSON object."
    )

    print(f"Generating {num_samples} training examples...")
    with open(output_file, "w") as f:
        for _ in range(num_samples):
            user_prompt, target_json = generate_random_example()

            # OpenAI / HuggingFace standard ChatML format
            chat_example = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": json.dumps(target_json)},
                ]
            }
            f.write(json.dumps(chat_example) + "\n")

    print(f"Dataset written to {output_file}.")
    print("\nNext Steps for Fine-Tuning:")
    print("1. Upload this file to HuggingFace or a local Jupyter Notebook.")
    print("2. Load a base model (e.g., 'microsoft/Phi-3-mini-4k-instruct').")
    print("3. Apply LoRA using the 'peft' library.")
    print("4. Train for 2-3 epochs using 'SFTTrainer'.")
    print("5. Export and quantize to GGUF via llama.cpp for use in synth-pdb's local backend!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SLM training data.")
    parser.add_argument("--samples", type=int, default=100, help="Number of examples to generate.")
    parser.add_argument("--output", type=str, default="train.jsonl", help="Output filename.")
    args = parser.parse_args()

    build_huggingface_dataset(args.samples, args.output)
