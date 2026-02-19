#!/bin/bash
# Test all demo scripts work on headless VM (no display required)
#
# Usage: bash test_all_demos.sh

echo "=========================================="
echo "Testing All Demo Scripts (Headless Mode)"
echo "=========================================="
echo ""

DEMO_DIR="$(dirname "$0")"
cd "$DEMO_DIR"

PASSED=0
FAILED=0
FAILED_SCRIPTS=""

# Test each demo script
for script in *.py; do
    # Skip test script itself and __init__.py
    if [[ "$script" == "test_all_demos.py" || "$script" == "__init__.py" ]]; then
        continue
    fi

    echo "----------------------------------------"
    echo "Testing: $script"
    echo "----------------------------------------"

    # Run with timeout and capture output
    if timeout 120 python "$script" > /tmp/demo_output.txt 2>&1; then
        echo "✓ PASSED: $script"
        PASSED=$((PASSED + 1))
    else
        EXIT_CODE=$?
        echo "✗ FAILED: $script (exit code: $EXIT_CODE)"
        echo "Last 20 lines of output:"
        tail -n 20 /tmp/demo_output.txt
        FAILED=$((FAILED + 1))
        FAILED_SCRIPTS="$FAILED_SCRIPTS\n  - $script"
    fi
    echo ""
done

# Summary
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Passed: $PASSED"
echo "Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed scripts:$FAILED_SCRIPTS"
    echo ""
    exit 1
else
    echo ""
    echo "✓ All demos passed!"
    echo ""
    exit 0
fi
