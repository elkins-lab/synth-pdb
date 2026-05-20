import argparse
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from synth_pdb.llm import LLMInterface, OpenAILLMProvider, LocalLLMProvider


class TestLLMInterface(unittest.TestCase):
    """Test suite for the LLM Prompt-to-Protein Interface."""

    def setUp(self):
        # Ensure environment variables are clean for testing
        self.env_patcher = patch.dict(os.environ, {"OPENAI_API_KEY": "fake-test-key"}, clear=True)
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_openai_missing_api_key(self):
        """Test that missing API key raises an appropriate error for openai backend."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                ValueError, "OPENAI_API_KEY environment variable is missing"
            ):
                LLMInterface(backend="openai")

    def test_json_parsing_clean(self):
        """Test parsing of a clean JSON response."""
        llm = LLMInterface(backend="openai")
        clean_json = '{"length": 20, "cyclic": true}'
        parsed = llm.provider._parse_response(clean_json)
        self.assertEqual(parsed, {"length": 20, "cyclic": True})

    def test_json_parsing_markdown(self):
        """Test parsing of JSON embedded in markdown code blocks."""
        llm = LLMInterface(backend="openai")
        markdown_json = """
Here is your configuration:
```json
{
    "sequence": "ALAGLY",
    "minimize": true,
    "metal_ions": "auto"
}
```
Good luck!
"""
        parsed = llm.provider._parse_response(markdown_json)
        self.assertEqual(parsed, {"sequence": "ALAGLY", "minimize": True, "metal_ions": "auto"})

    def test_json_parsing_invalid(self):
        """Test parsing of invalid JSON raises RuntimeError."""
        llm = LLMInterface(backend="openai")
        invalid_json = '{"sequence": "ALAGLY", "minimize": true'  # Missing closing brace
        with self.assertRaisesRegex(RuntimeError, "Failed to parse LLM response as JSON"):
            llm.provider._parse_response(invalid_json)

    @patch("synth_pdb.llm.urllib.request.urlopen")
    def test_openai_translate_prompt_success(self, mock_urlopen):
        """Test a successful API call and translation using OpenAI backend."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response_data = {
            "choices": [
                {
                    "message": {
                        "content": '{"length": 15, "conformation": "beta", "cap_termini": true}'
                    }
                }
            ]
        }
        mock_response.read.return_value = json.dumps(mock_response_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        llm = LLMInterface(backend="openai")
        result = llm.translate_prompt("Make a 15 residue beta sheet and cap it")

        self.assertEqual(result, {"length": 15, "conformation": "beta", "cap_termini": True})

        request_obj = mock_urlopen.call_args[0][0]
        self.assertEqual(request_obj.headers.get("Authorization"), "Bearer fake-test-key")

    @patch("synth_pdb.llm.urllib.request.urlopen")
    def test_openai_translate_prompt_http_error(self, mock_urlopen):
        """Test handling of HTTP errors from the API."""
        from urllib.error import HTTPError

        mock_urlopen.side_effect = HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )

        llm = LLMInterface(backend="openai")
        with self.assertRaisesRegex(
            RuntimeError, "LLM API request failed: HTTP Error 401: Unauthorized"
        ):
            llm.translate_prompt("Generate something")

    # To mock local LLM dependencies properly without them installed
    @patch.dict("sys.modules", {"llama_cpp": MagicMock(), "huggingface_hub": MagicMock()})
    def test_local_translate_prompt_success(self):
        """Test a successful translation using Local LLM backend."""
        import llama_cpp
        import huggingface_hub

        # Setup mocks
        mock_hf_download = huggingface_hub.hf_hub_download
        mock_hf_download.return_value = "/fake/path/to/model.gguf"

        mock_llama_instance = MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '{"length": 30, "cyclic": true}'}}]
        }
        llama_cpp.Llama.return_value = mock_llama_instance

        # We need to reload the module or just instantiate because LLMInterface handles import internally
        llm = LLMInterface(backend="local")

        # Verify it downloaded
        mock_hf_download.assert_called_once_with(
            repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
            filename="Phi-3-mini-4k-instruct-q4.gguf",
        )

        result = llm.translate_prompt("Generate a 30mer cyclic")
        self.assertEqual(result, {"length": 30, "cyclic": True})

    @patch("synth_pdb.main.generate_pdb_content")
    @patch("sys.argv", ["synth-pdb", "--prompt", "Make an alpha helix", "--llm-backend", "openai"])
    @patch("synth_pdb.llm.LLMInterface")
    def test_main_integration_with_prompt(self, MockLLMInterface, mock_generate):
        """Test that main.py intercepts --prompt, calls the LLM, and updates args."""
        # Setup mock LLM
        mock_llm_instance = MagicMock()
        mock_llm_instance.translate_prompt.return_value = {"length": 10, "conformation": "alpha"}
        MockLLMInterface.return_value = mock_llm_instance

        mock_generate.return_value = "ATOM 1..."

        from synth_pdb.main import main

        with patch("builtins.open", MagicMock()):
            try:
                main()
            except SystemExit:
                pass

        MockLLMInterface.assert_called_once_with(backend="openai")
        mock_llm_instance.translate_prompt.assert_called_once_with("Make an alpha helix")
        self.assertTrue(mock_generate.called)

    def test_local_llm_missing_deps(self):
        """Verify that LocalLLMProvider raises ImportError when llama_cpp is missing."""
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with self.assertRaisesRegex(ImportError, "Local LLM support requires additional"):
                LocalLLMProvider()
