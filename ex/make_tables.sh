#!/bin/bash

# Define paths to the scripts
script1="./Table1.sh"
script2="./Table2.sh"
script3="./Table3.sh"

# Initialize total execution time
total_start=$(date +%s)

# Function to execute a script and measure its execution time
run_and_time() {
    local script_path=$1
    local start_time=$(date +%s)
    
    echo "Running $script_path..."
    
    bash "$script_path"
    
    local end_time=$(date +%s)
    local execution_time=$((end_time - start_time))
    echo "$script_path took $execution_time seconds."
    echo "---------------------------------------"
    
    # Return execution time
    echo $execution_time
}

cd benchmarks-deterministic

# Run each script and record their individual execution times
time1=$(run_and_time "$script1")

cd ../benchmarks-stochastic

time2=$(run_and_time "$script2")
time3=$(run_and_time "$script3")

# Calculate total execution time
total_end=$(date +%s)
total_time=$((total_end - total_start))

cd ..

cp ./benchmarks-deterministic/Table1.csv .
cp ./benchmarks-stochastic/Table2.csv .
cp ./benchmarks-stochastic/Table3.csv .

# Output individual times and total time
echo "Execution times for each table:"
echo "Table1.sh: $time1 seconds"
echo "Table2.sh: $time2 seconds"
echo "Table3.sh: $time3 seconds"
echo "Total execution time for all scripts: $total_time seconds"
