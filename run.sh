#!/bin/bash

# Number of times to run the Python script
num_runs=1000000

# Counter for successful runs
success_count=0

# Maximum number of concurrent processes
max_concurrent=5

# Loop to execute the Python script in parallel threads with a delay
for ((i=1; i<=$num_runs; i++)); do
    # Start a new process in the background and capture the output
    output=$(python ai.py 2>&1 &)

    # Increment the success count
    ((success_count++))
    echo "Run $i started with output: $output"

    # Check if the number of background processes exceeds the limit
    if ((i % max_concurrent == 0)); then
        # Wait for all background processes to finish before starting the next batch
        wait
    fi

    # Introduce a delay of 1 second
    sleep 1
done

# Wait for any remaining background processes to finish
wait

# Display the count of successful runs
echo "Successful runs: $success_count"

# Exit the bash script
exit 0
