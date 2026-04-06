#!/bin/bash

# find_breaking_changes.sh
# Script to identify which commit or uncommitted changes broke the tests.

# Exit immediately if a pipeline, which may consist of a single simple command, 
# a list, or a compound command returns a non-zero status.
# We will temporarily disable this around pytest runs.
set -e

# Ensure we can return to the exact same state
ORIGINAL_HEAD=$(git rev-parse HEAD)
ORIGINAL_BRANCH=$(git branch --show-current)
STASHED=0
STASH_NAME=""

# Function to restore original state safely
restore_state() {
    echo ""
    echo "--- Restoring Original State ---"
    
    # Checkout original branch or commit
    if [ -n "$ORIGINAL_BRANCH" ]; then
        git checkout -q "$ORIGINAL_BRANCH"
    else
        git checkout -q "$ORIGINAL_HEAD"
    fi
    
    # Restore stashed changes if we stashed them
    if [ $STASHED -eq 1 ]; then
        echo "Restoring your uncommitted changes from stash ($STASH_NAME)..."
        # Find the specific stash by name
        STASH_REF=$(git stash list | grep "$STASH_NAME" | cut -d: -f1 | head -n 1)
        if [ -n "$STASH_REF" ]; then
            git stash apply "$STASH_REF" >/dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo "✅ Successfully applied uncommitted changes."
                echo "Note: The stash '$STASH_NAME' has been kept as a backup just in case."
                echo "You can drop it with: git stash drop $STASH_REF"
            else
                echo "⚠️ Warning: Failed to apply stash (possible conflicts)."
                echo "Your changes are safely stored in: $STASH_REF ($STASH_NAME)"
            fi
        else
            echo "⚠️ Warning: Could not find stash reference for $STASH_NAME."
            echo "Please check 'git stash list' manually."
        fi
    fi
}

# Trap script termination (Ctrl+C) to ensure we always restore state
trap restore_state SIGINT SIGTERM

echo "Step 1: Running initial pytest..."
set +e
pytest
TEST_STATUS=$?
set -e

if [ $TEST_STATUS -eq 0 ]; then
    echo "✅ Success: All tests passed! No breakages found."
    exit 0
fi

echo ""
echo "❌ Tests failed in the current state. Investigating cause..."

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo ""
    echo "Step 2: Uncommitted changes detected. Shelving them safely..."
    STASH_NAME="test_failure_stash_$(date +%Y%m%d_%H%M%S)"
    # Push changes including untracked files
    git stash push -u -m "$STASH_NAME"
    STASHED=1
    
    echo "Running pytest without uncommitted changes..."
    set +e
    pytest
    STASH_TEST_STATUS=$?
    set -e
    
    if [ $STASH_TEST_STATUS -eq 0 ]; then
        echo ""
        echo "✅ Success without uncommitted changes."
        echo "======================================================"
        echo "🔍 ANALYSIS REPORT"
        echo "======================================================"
        echo "Your recent UNCOMMITTED changes are causing the tests to fail."
        echo "The last commit is completely fine."
        echo "======================================================"
        restore_state
        exit 1
    fi
    echo ""
    echo "❌ Tests still fail without uncommitted changes."
    echo "The breakage is present in the current commit."
else
    echo "No uncommitted changes detected."
fi

echo ""
echo "Step 3: Searching commit history to find the breaking commit..."
echo "Going back in history until tests pass (checking up to 50 commits)..."

# Use git log to check history. (Using standard git instead of gh for local robustness)
COMMITS=$(git log --format="%H" -n 50)
PREV_COMMIT=""
FOUND_GOOD=0

for COMMIT in $COMMITS; do
    echo "Checking out commit ${COMMIT:0:7}..."
    git checkout -q "$COMMIT"
    
    # Run pytest quietly to avoid console spam during the search
    set +e
    pytest -q --disable-warnings > /dev/null 2>&1
    COMMIT_TEST_STATUS=$?
    set -e
    
    if [ $COMMIT_TEST_STATUS -eq 0 ]; then
        echo "✅ Found last good commit: ${COMMIT:0:7}"
        FOUND_GOOD=1
        break
    fi
    PREV_COMMIT=$COMMIT
done

# We have finished searching, restore the user's workspace
# Disable trap since we're calling restore manually now
trap - SIGINT SIGTERM
restore_state

echo ""
echo "======================================================"
echo "🔍 ANALYSIS REPORT"
echo "======================================================"

if [ $FOUND_GOOD -eq 1 ]; then
    if [ -n "$PREV_COMMIT" ]; then
        echo "🔴 The tests started failing at commit: ${PREV_COMMIT:0:7}"
        echo "🟢 Last known good commit: ${COMMIT:0:7}"
        echo ""
        echo "Details of the breaking commit:"
        git log -1 "$PREV_COMMIT"
    else
        echo "The current commit is good, but this shouldn't be reached if tests failed initially."
    fi
else
    echo "❌ Could not find a passing commit in the last 50 commits."
    echo "The breakage might be older, or related to your local environment/dependencies."
    echo "Try increasing the limit in the script or check environment configuration."
fi
echo "======================================================"

exit 1