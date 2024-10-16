#!/bin/bash

# Directory containing Python files (change this to your target directory)
PYTHON_DIR="Table 2 - Single Runs"
# Output file for the table
OUTPUT_FILE="Table2.csv"

# Initialize the output file with headers
echo "File,Gamma,Lambda,c,Confidence,Execution Time (seconds)" > $OUTPUT_FILE

# Loop over each Python file in the specified directory
for pyfile in "$PYTHON_DIR"/*.py; do
    # Get the filename without the path
    filename=$(basename "$pyfile")
    
    # Start time
    #start_time=$(date +%s.%N)
    
    # Execute the Python file and capture the output
    output=$(python3 "$pyfile" 2>&1)
    
    # End time and calculate execution time
    #end_time=$(date +%s.%N)
    #execution_time=$(echo "$end_time - $start_time" | bc)
    execution_time=$(echo "$output" | grep -oP "(?<=time: )\S+" || echo "N/A")
    # Extract gamma, lambda, and other values from output
    # Assuming gamma and lambda are printed in the form "gamma: value" and "lambda: value"
    
    gamma=$(echo "$output" | grep -oP "(?<='gamma': )\S+" || echo "N/A")
    lambda=$(echo "$output" | grep -oP "(?<='lambda': )\S+" || echo "N/A")
    c=$(echo "$output" | grep -oP "(?<='c': )\S+" || echo "N/A")
    confidence=$(echo "$output" | grep -oP "(?<='confidence': )\S+" || echo "N/A")
    
    # Remove any trailing '}' from lambda or execution time if they exist
    lambda=$(echo "$lambda" | sed 's/}$//')
    c=$(echo "$c" | sed 's/}$//')
    confidence=$(echo "$confidence" | sed 's/}$//')
    execution_time=$(echo "$execution_time" | sed 's/}$//')

    
    echo "Filename: $filename"
    echo "Gamma: $gamma"
    echo "Lambda: $lambda"
    echo "c: $c"
    echo "Confidence: $confidence"
    echo "Execution Time: $execution_time seconds"
    echo "-----------------------------"
    
    
    # Append the results to the CSV file
    echo "\"$filename\",\"$gamma\",\"$lambda\",\"$c\",\"$confidence\",\"$execution_time\"" >> $OUTPUT_FILE
done

echo "Sorting Rows to Match Paper..."

# Define the file and row numbers
FILE="Table2.csv"
ROW_TO_MOVE=7
TARGET_POSITION=3

# Store the row to move in a temporary variable
ROW_CONTENT=$(sed -n "${ROW_TO_MOVE}p" "$FILE")

# Remove the row from the file
sed -i "${ROW_TO_MOVE}d" "$FILE"

# Insert the row content at the target position +1 (to insert after the target)
awk -v line="$ROW_CONTENT" -v pos=$((TARGET_POSITION+1)) 'NR==pos {print line} 1' "$FILE" > temp_file && mv temp_file "$FILE"


echo "Execution complete. Results are stored in $OUTPUT_FILE."

