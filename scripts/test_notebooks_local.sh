#!/bin/bash

VENV_NAME=".venv_colab"

if [ ! -d "$VENV_NAME" ]; then
    echo "Error: Virtual environment $VENV_NAME not found."
    echo "Please run scripts/setup_colab_venv.sh first."
    exit 1
fi

echo "Activating $VENV_NAME..."
source "$VENV_NAME/bin/activate"

# Set Test Mode to skip long-running computations in notebooks
export SYNTH_PDB_TEST_MODE=1

# Create a temporary directory for output notebooks
NB_OUT_DIR="/tmp/synth_pdb_nb_out"
mkdir -p "$NB_OUT_DIR"

echo "Running notebook tests locally..."
echo "Output notebooks saved to: $NB_OUT_DIR"
echo "------------------------------------------------"

# List of directories containing notebooks to test
NOTEBOOK_DIRS=(
    "examples/interactive_tutorials"
    "examples/ml_integration"
    "examples/ml_loading"
)

PASSED=()
FAILED=()

# Iterate through each directory
for dir in "${NOTEBOOK_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Checking directory: $dir"
        
        # Iterate through each .ipynb file in the directory
        for nb in "$dir"/*.ipynb; do
            # Skip if no notebooks found (glob returns pattern if no match)
            [ -e "$nb" ] || continue
            
            nb_name=$(basename "$nb")
            echo -n "  Testing $nb_name... "
            
            # Execute the notebook
            # We redirect stderr to a log file to keep the console clean but save errors
            LOG_FILE="$NB_OUT_DIR/${nb_name}.log"
            
            if jupyter nbconvert --to notebook --execute \
                --output-dir="$NB_OUT_DIR" \
                --ExecutePreprocessor.timeout=600 \
                "$nb" > "$LOG_FILE" 2>&1; then
                echo "✅ PASSED"
                PASSED+=("$nb")
            else
                echo "❌ FAILED"
                FAILED+=("$nb")
                # Print the last few lines of the error log for immediate feedback
                echo "     Summary of error (see $LOG_FILE for full trace):"
                tail -n 10 "$LOG_FILE" | sed 's/^/     /'
            fi
        done
    fi
done

echo "------------------------------------------------"
echo "TEST SUMMARY"
echo "  Total:  $((${#PASSED[@]} + ${#FAILED[@]}))"
echo "  Passed: ${#PASSED[@]}"
echo "  Failed: ${#FAILED[@]}"

if [ ${#FAILED[@]} -ne 0 ]; then
    echo ""
    echo "FAILED NOTEBOOKS:"
    for nb in "${FAILED[@]}"; do
        echo "  - $nb"
    done
    exit 1
else
    echo ""
    echo "All notebooks completed successfully! 🎉"
    exit 0
fi
