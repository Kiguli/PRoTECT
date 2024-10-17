#!/bin/bash

# Directories containing Python files
PYTHON_DIR="./PRoTECT-versions"
FOSSIL_DIR="./FOSSIL-versions"

# Output file for the table
OUTPUT_FILE="Table1.csv"

# Initialize the output file with headers
echo "File,Gamma,Lambda,PRoTECT (secs),FOSSIL (secs)" > $OUTPUT_FILE

# Loop over each Python file in the specified directory
for pyfile in "$PYTHON_DIR"/*.py; do
    # Get the filename without the path
    filename=$(basename "$pyfile")
    
    # Execute the Python file in PRoTECT-versions and capture the output
    output=$(python3 "$pyfile" 2>&1)
    
    # Extract gamma, lambda, and execution time
    gamma=$(echo "$output" | grep -oP "(?<='gamma': )\S+" || echo "N/A")
    lambda=$(echo "$output" | grep -oP "(?<='lambda': )\S+" || echo "N/A")
    execution_time_protect=$(echo "$output" | grep -oP "(?<=time: )\S+" || echo "N/A")
    
    # Clean up extracted values
    lambda=$(echo "$lambda" | sed 's/}$//')
    execution_time_protect=$(echo "$execution_time_protect" | sed 's/}$//')
    
    # Initialize execution_time_fossil as "N/A" by default
    execution_time_fossil="N/A"
    
    # Check if a corresponding file exists in FOSSIL-versions
    fossil_file="$FOSSIL_DIR/$filename"
    if [[ -f "$fossil_file" ]]; then
        # Execute the Python file in FOSSIL-versions and capture the output
        output_fossil=$(python3 "$fossil_file" 2>&1)
        # Extract the execution time for FOSSIL
        execution_time_fossil=$(echo "$output_fossil" | grep -oP "(?<=Time: )\S+" || echo "N/A")
        # Clean up extracted value
        execution_time_fossil=$(echo "$execution_time_fossil" | sed 's/}$//')
    fi
    
    echo "Filename: $filename"
    echo "Gamma: $gamma"
    echo "Lambda: $lambda"
    echo "Execution Time (PRoTECT): $execution_time_protect seconds"
    echo "Execution Time (FOSSIL): $execution_time_fossil seconds"
    echo "-----------------------------"
    
    # Append the results to the CSV file
    echo "\"$filename\",\"$gamma\",\"$lambda\",\"$execution_time_protect\",\"$execution_time_fossil\"" >> $OUTPUT_FILE
done

echo "Sorting Rows to Match Paper..."

# Define the file and row numbers
FILE="Table1.csv"
ROW_TO_MOVE=5
TARGET_POSITION=3

# Store the row to move in a temporary variable
ROW_CONTENT=$(sed -n "${ROW_TO_MOVE}p" "$FILE")

# Remove the row from the file
sed -i "${ROW_TO_MOVE}d" "$FILE"

# Insert the row content at the target position +1 (to insert after the target)
awk -v line="$ROW_CONTENT" -v pos=$((TARGET_POSITION+1)) 'NR==pos {print line} 1' "$FILE" > temp_file && mv temp_file "$FILE"

echo "Execution complete. Results are stored in $OUTPUT_FILE."

