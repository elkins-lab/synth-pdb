"""
CI startup script injected into every notebook kernel before execution.

Mocks platform-specific modules that are not available outside their
native runtime, so notebooks that guard these with try/except still
import cleanly, and notebooks that import unconditionally don't crash CI.

Injected via:
  --ExecutePreprocessor.extra_arguments='["--IPKernelApp.exec_files=scripts/ci_notebook_startup.py"]'
"""
import sys
from unittest.mock import MagicMock

# --- google.colab ---
# Notebooks use google.colab for file-upload widgets and auth flows.
# These UI features are untestable outside Colab; mock them away so the
# rest of the notebook can execute normally.
google_mock = MagicMock()
sys.modules.setdefault("google", google_mock)
sys.modules.setdefault("google.colab", google_mock.colab)
sys.modules.setdefault("google.colab.files", google_mock.colab.files)
sys.modules.setdefault("google.colab.output", google_mock.colab.output)
