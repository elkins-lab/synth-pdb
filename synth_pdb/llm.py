import json
import os
import re
import urllib.request
from typing import Any, Dict, Optional, cast
from urllib.error import HTTPError, URLError

SYSTEM_PROMPT = """You are a CLI argument translator for synth-pdb, a synthetic protein generator.
Your ONLY job is to output a JSON object mapping argument names to their values based on the user's request.
DO NOT attempt to perform the biological task yourself. DO NOT generate chemical shifts or peptide sequences yourself.

Output a JSON object using ONLY these allowed keys:
{
  "length": <int>,
  "sequence": <str>,
  "conformation": <"alpha"|"beta"|"ppii"|"extended"|"random">,
  "minimize": <bool>,
  "cap_termini": <bool>,
  "cyclic": <bool>,
  "metal_ions": <"auto"|"none">,
  "output": <str>,
  "gen_shifts": <bool>,
  "gen_relax": <bool>,
  "gen_cd": <bool>
}

Example Request: "Build me a 20-residue alpha helix, make it a cyclic peptide, and generate the NMR chemical shifts for it."
Example Output: {"length": 20, "conformation": "alpha", "cyclic": true, "gen_shifts": true}

Output ONLY the raw JSON object. No other text.
"""


class LLMProvider:
    """Abstract base class for LLM Providers."""

    def translate_prompt(self, prompt: str) -> dict:
        raise NotImplementedError

    def _parse_response(self, text: str) -> dict:
        """Extract and parse JSON from the LLM response."""
        text = text.strip()
        # Remove markdown code blocks if present
        json_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
        match = json_pattern.search(text)
        if match:
            text = match.group(1)

        try:
            return cast(dict, json.loads(text))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse LLM response as JSON: {e}\nResponse was: {text}")


class OpenAILLMProvider(LLMProvider):
    """Uses urllib to call an OpenAI-compatible REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1/chat/completions",
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is missing. Please set it to use the openai backend."
            )
        self.base_url = base_url
        self.model = model

    def translate_prompt(self, prompt: str) -> dict:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }

        req = urllib.request.Request(
            self.base_url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST"
        )

        try:
            with urllib.request.urlopen(req) as response:
                response_body = response.read().decode("utf-8")
                response_json = json.loads(response_body)
                content = response_json["choices"][0]["message"]["content"]

                parsed_args = self._parse_response(content)
                if "error" in parsed_args:
                    raise ValueError(f"LLM could not translate prompt: {parsed_args['error']}")
                return parsed_args

        except HTTPError as e:
            raise RuntimeError(f"LLM API request failed: HTTP Error {e.code}: {e.reason}")
        except URLError as e:
            raise RuntimeError(f"LLM API request failed: URL Error: {e.reason}")


class LocalLLMProvider(LLMProvider):
    """Uses llama-cpp-python to run a local SLM (e.g. Phi-3-mini) for zero-shot translation."""

    def __init__(
        self,
        repo_id: str = "microsoft/Phi-3-mini-4k-instruct-gguf",
        filename: str = "Phi-3-mini-4k-instruct-q4.gguf",
    ):
        try:
            import llama_cpp
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "Local LLM support requires additional dependencies. "
                "Please install them via: pip install synth-pdb[local-llm]"
            )

        print(f"Loading local model '{filename}' (will download if not cached)...")
        self.model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        # Initialize llama.cpp
        # n_gpu_layers=-1 attempts to offload all layers to Metal/CUDA
        self.llm = llama_cpp.Llama(
            model_path=self.model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False
        )

    def translate_prompt(self, prompt: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        schema = {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "length": {"type": "integer"},
                    "sequence": {"type": "string"},
                    "conformation": {
                        "type": "string",
                        "enum": ["alpha", "beta", "ppii", "extended", "random"],
                    },
                    "minimize": {"type": "boolean"},
                    "cap_termini": {"type": "boolean"},
                    "cyclic": {"type": "boolean"},
                    "metal_ions": {"type": "string", "enum": ["auto", "none"]},
                    "output": {"type": "string"},
                    "gen_shifts": {"type": "boolean"},
                    "gen_relax": {"type": "boolean"},
                    "gen_cd": {"type": "boolean"},
                },
            },
        }

        response = self.llm.create_chat_completion(
            messages=messages, temperature=0.0, max_tokens=256, response_format=schema
        )

        content = response["choices"][0]["message"]["content"]
        parsed_args = self._parse_response(content)
        if "error" in parsed_args:
            raise ValueError(f"Local LLM could not translate prompt: {parsed_args['error']}")
        return parsed_args


class LLMInterface:
    """Factory and wrapper for backward compatibility."""

    def __init__(self, backend: str = "local", **kwargs: Any) -> None:
        self.provider: LLMProvider
        if backend == "openai":
            self.provider = OpenAILLMProvider(**kwargs)
        elif backend == "local":
            self.provider = LocalLLMProvider(**kwargs)
        else:
            raise ValueError(f"Unknown LLM backend: {backend}")

    def translate_prompt(self, prompt: str) -> dict:
        return self.provider.translate_prompt(prompt)
