#!/bin/bash
set -e

# Configuration
VENV_NAME=".venv_colab"
PYTHON_BIN=$(which python3)

echo "=== Setting up Colab-like Virtual Environment ==="
echo "Using Python: $PYTHON_BIN"

# 1. Create virtual environment
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment in $VENV_NAME..."
    $PYTHON_BIN -m venv "$VENV_NAME"
else
    echo "Virtual environment $VENV_NAME already exists."
fi

# 2. Activate environment
source "$VENV_NAME/bin/activate"

# 3. Upgrade core tools
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# 4. Install Colab Baseline Packages
# These are packages commonly pre-installed in Google Colab runtimes
echo "Installing Colab baseline packages..."
pip install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scipy \
    scikit-learn \
    ipykernel \
    nbconvert \
    ipywidgets \
    tqdm \
    requests \
    altair \
    bokeh \
    plotly \
    statsmodels \
    umap-learn \
    pillow \
    py3Dmol \
    pynmrstar \
    jax \
    jaxlib \
    "openmm>=8.1.0,!=8.5.0,!=8.5.1"

# 5. Install synth-pdb with all optional extras
echo "Installing synth-pdb with [dev,ai,gnn,plm,test] extras..."
pip install -e ".[dev,ai,gnn,plm,test]"

# 5b. Install local sibling packages (private, not on PyPI)
# These packages are required by interactive tutorials but are not on PyPI.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SIBLING_PKGS="synth-saxs synth-dynamics synth-afm"
for pkg in $SIBLING_PKGS; do
    PKG_DIR="$(dirname "$REPO_ROOT")/$pkg"
    if [ -d "$PKG_DIR" ]; then
        echo "Installing local $pkg from $PKG_DIR..."
        pip install -e "$PKG_DIR"
    else
        echo "Warning: $pkg not found at $PKG_DIR. Some notebooks may fail."
        echo "         Clone it alongside this repo: git clone <$pkg-repo>"
    fi
done

# 6. Verify installation
echo "Checking installation..."
python -c "import synth_pdb; print(f'synth-pdb version: {synth_pdb.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo "To activate this environment, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To test your notebooks in this environment, you can run:"
echo "  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 examples/interactive_tutorials/*.ipynb"
