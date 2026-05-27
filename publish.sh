#!/bin/bash
# publish.sh - Clean, build, and publish synth-pdb to PyPI
# Usage: ./publish.sh [test|prod]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Determine target (test or production PyPI)
TARGET="${1:-prod}"

echo -e "${YELLOW}🧬 synth-pdb PyPI Publisher${NC}"
echo "================================"
echo ""

# Step 1: Check current version
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
echo -e "${GREEN}✓${NC} Current version: ${CURRENT_VERSION}"
echo ""

# Step 1.5: NumPy 2.0 Compatibility Check
echo -e "${YELLOW}Checking environment compatibility...${NC}"
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "0.0.0")
if [[ "$NUMPY_VERSION" == 2.* ]]; then
    echo -e "${RED}⚠️  NumPy 2.x detected ($NUMPY_VERSION)${NC}"
    echo "This project requires NumPy 1.x to avoid C-extension crashes in dependencies like Numba and PyArrow."
    echo ""
    read -p "Would you like to downgrade to NumPy 1.x now? (yes/no): " FIX_NUMPY
    if [ "$FIX_NUMPY" = "yes" ]; then
        pip install "numpy<2.0.0"
        echo -e "${GREEN}✓${NC} NumPy downgraded."
    else
        echo -e "${RED}✗${NC} Incompatible NumPy version. Aborting to prevent crash."
        exit 1
    fi
fi
echo -e "${GREEN}✓${NC} Environment compatible."
echo ""

# Step 2: Run tests
echo -e "${YELLOW}Running tests...${NC}"
if pytest tests/ --tb=no -q; then
    echo -e "${GREEN}✓${NC} All tests passed!"
else
    echo -e "${RED}✗${NC} Tests failed! Aborting."
    exit 1
fi
echo ""

# Step 3: Clean old builds
echo -e "${YELLOW}Cleaning old builds...${NC}"
rm -rf dist/
rm -rf build/
rm -rf *.egg-info
echo -e "${GREEN}✓${NC} Cleaned dist/, build/, and *.egg-info"
echo ""

# Step 4: Build package
echo -e "${YELLOW}Building package...${NC}"
python -m build
echo -e "${GREEN}✓${NC} Package built successfully"
echo ""

# Step 5: List built files
echo -e "${YELLOW}Built files:${NC}"
ls -lh dist/
echo ""

# Step 6: Upload to PyPI
if [ "$TARGET" = "test" ]; then
    echo -e "${YELLOW}Uploading to TestPyPI...${NC}"
    python -m twine upload --repository testpypi dist/*
    echo ""
    echo -e "${GREEN}✓${NC} Uploaded to TestPyPI!"
    echo ""
    echo "Test installation with:"
    echo "  pip install --index-url https://test.pypi.org/simple/ --no-deps synth-pdb"
elif [ "$TARGET" = "prod" ]; then
    echo -e "${YELLOW}⚠️  About to upload to PRODUCTION PyPI${NC}"
    echo "Version: ${CURRENT_VERSION}"
    echo ""
    read -p "Are you sure? (yes/no): " CONFIRM

    if [ "$CONFIRM" = "yes" ]; then
        echo ""
        echo -e "${YELLOW}Uploading to PyPI...${NC}"
        python -m twine upload dist/*
        echo ""
        echo -e "${GREEN}✓${NC} Successfully published to PyPI!"
        echo ""
        echo "Install with:"
        echo "  pip install synth-pdb"
        echo ""
        echo "View at:"
        echo "  https://pypi.org/project/synth-pdb/${CURRENT_VERSION}/"
    else
        echo -e "${RED}✗${NC} Upload cancelled."
        exit 1
    fi
else
    echo -e "${RED}✗${NC} Invalid target: $TARGET"
    echo "Usage: ./publish.sh [test|prod]"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 Done!${NC}"
