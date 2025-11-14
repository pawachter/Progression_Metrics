#!/bin/bash

# Script to run main.py multiple times
# Each run will create its own timestamped log directory

NUM_RUNS=50
SCRIPT_PATH="main.py"

echo "=========================================="
echo "Starting $NUM_RUNS training runs"
echo "=========================================="
echo ""

for i in $(seq 1 $NUM_RUNS); do
    echo "=========================================="
    echo "Starting Run $i of $NUM_RUNS"
    echo "=========================================="
    
    # Run the Python script
    python "$SCRIPT_PATH"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Run $i completed successfully"
        echo ""
    else
        echo ""
        echo "✗ Run $i failed with exit code $EXIT_CODE"
        echo "Do you want to continue? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Stopping execution at run $i"
            exit 1
        fi
    fi
    
    # Small delay between runs
    sleep 2
done

echo ""
echo "=========================================="
echo "All $NUM_RUNS runs completed!"
echo "=========================================="
echo "Check the logs/ directory for results from each run"
