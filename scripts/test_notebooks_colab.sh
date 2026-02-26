#!/bin/bash
set -e

echo "Simulating Google Colab Environment for Notebook Testing..."

# The official Colab CPU runtime image
COLAB_IMAGE="us-docker.pkg.dev/colab-images/public/cpu-runtime"

# ---------------------------------------------------------------------------
# Runtime detection: prefer Colima (lightweight, avoids Docker Desktop issues)
# then fall back to Docker Desktop.
# ---------------------------------------------------------------------------
if command -v colima &> /dev/null; then
    echo "Colima detected. Using Colima as the container runtime."

    # Start Colima if it isn't already running
    if ! colima status &> /dev/null 2>&1; then
        echo "Starting Colima (CPU=4, Memory=8GB)..."
        colima start --cpu 4 --memory 8
    else
        echo "Colima is already running."
    fi

    # Point the Docker CLI at Colima's socket
    export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"
elif command -v docker &> /dev/null; then
    echo "Colima not found. Falling back to Docker Desktop."
    # Ensure Docker Desktop daemon is actually running
    if ! docker info &> /dev/null 2>&1; then
        echo "Error: Docker Desktop does not appear to be running. Please start Docker Desktop and try again."
        exit 1
    fi
else
    echo "Error: Neither Colima nor Docker could be found."
    echo "  Install Colima (recommended): brew install colima docker"
    echo "  Or install Docker Desktop:    https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Use a temporary docker config to avoid "docker-credential-osxkeychain" errors on Mac
export DOCKER_CONFIG=$(mktemp -d)
echo "{}" > "$DOCKER_CONFIG/config.json"

echo "Pulling the latest Colab CPU runtime image (this may take a while the first time)..."
docker pull $COLAB_IMAGE

# Get the absolute path of the repository root
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

echo "Running notebook tests inside the Colab container..."
docker run --rm -v "${REPO_ROOT}:/workspace" -w /workspace $COLAB_IMAGE /bin/bash -c "
echo 'Installing package and dependencies...' &&
pip install -e '.[dev,plm,gnn,ai]' &&
pip install umap-learn matplotlib seaborn &&
echo 'Running notebook tests with jupyter nbconvert...' &&
jupyter nbconvert --to notebook --execute --output-dir=/tmp/nbout \
  --ExecutePreprocessor.timeout=600 \
  docs/tutorials/*.ipynb \
  examples/interactive_tutorials/*.ipynb \
  examples/ml_integration/*.ipynb \
  examples/ml_loading/*.ipynb
"

echo "Notebook testing in Colab simulation complete."
