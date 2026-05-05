import subprocess
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import io
import argparse

# Import the logic from main.py
# We can't easily import main() and run it without it trying to run the whole app.
# But we can test the logic if we wrap it or just use subprocess for integration tests.


def run_synth_pdb(args, stdin_input=None, env=None):
    """Helper to run synth-pdb CLI as a subprocess."""
    cmd = [sys.executable, "-m", "synth_pdb.main"] + args

    current_env = os.environ.copy()
    if env:
        current_env.update(env)
    current_env["OPENAI_API_KEY"] = "fake-key-for-testing"

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=current_env,
    )
    stdout, stderr = process.communicate(input=stdin_input, timeout=15)
    return process.returncode, stdout, stderr


class TestPromptStdin(unittest.TestCase):
    def test_prompt_piped_input(self):
        """Test that --prompt reads from piped stdin."""
        prompt_text = "Generate a 10 residue alpha helix"
        returncode, stdout, stderr = run_synth_pdb(
            ["--prompt", "--llm-backend", "openai", "--log-level", "DEBUG"], stdin_input=prompt_text
        )

        # Verify it read the prompt
        self.assertIn(f"Read prompt from stdin: {prompt_text}", stderr)
        # Verify it tried to call the LLM
        self.assertIn("LLM API request failed", stderr)

    def test_prompt_with_explicit_value_ignores_stdin(self):
        """Test that --prompt 'value' ignores stdin."""
        explicit_prompt = "Explicit prompt"
        piped_prompt = "Piped prompt"
        returncode, stdout, stderr = run_synth_pdb(
            ["--prompt", explicit_prompt, "--llm-backend", "openai", "--log-level", "DEBUG"],
            stdin_input=piped_prompt,
        )

        # Should NOT contain the piped prompt in the logs
        self.assertNotIn(f"Read prompt from stdin: {piped_prompt}", stderr)
        # Should show it translating the explicit one (or at least trying)
        # We can't check the exact translation because it fails 401, but we can check it didn't read stdin.

    def test_prompt_empty_stdin(self):
        """Test behavior when stdin is empty."""
        # Using a very short timeout to ensure it doesn't hang if it's waiting for input incorrectly
        returncode, stdout, stderr = run_synth_pdb(
            ["--prompt", "--llm-backend", "openai", "--log-level", "DEBUG"], stdin_input=""
        )
        # It shouldn't find a prompt
        self.assertNotIn("Read prompt from stdin:", stderr)


if __name__ == "__main__":
    unittest.main()
