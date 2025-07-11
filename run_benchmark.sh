#!/bin/bash

# =======================================================================
# VLURes Benchmark Execution Script
# =======================================================================
#
# This script runs the VLM inference for a specified language, task,
# and experimental setting.
#
# USAGE:
#   ./run_benchmark.sh <language> <task_number> <setting>
#
# ARGUMENTS:
#   <language>: The language to process (e.g., English, Japanese, Swahili, Urdu)
#   <task_number>: The task number to run (1-8)
#   <setting>: The experimental setting. Must be one of:
#              - zeroshot_no_rationales
#              - zeroshot_with_rationales
#              - oneshot_no_rationales
#              - oneshot_with_rationales
#
# EXAMPLE:
#   ./run_benchmark.sh English 1 zeroshot_no_rationales
#
# PREREQUISITES:
#   1. A Python environment with all dependencies from requirements.txt installed.
#   2. The OPENAI_API_KEY environment variable must be set:
#      export OPENAI_API_KEY="your-api-key"
# =======================================================================

# --- Argument Validation ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <language> <task_number> <setting>"
    echo "  <language>: English, Japanese, Swahili, or Urdu"
    echo "  <task_number>: 1-8"
    echo "  <setting>: zeroshot_no_rationales, zeroshot_with_rationales, oneshot_no_rationales, oneshot_with_rationales"
    exit 1
fi

LANGUAGE=$1
TASK_NUM=$2
SETTING=$3

# --- Check for API Key ---
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: The OPENAI_API_KEY environment variable is not set."
    echo "Please set it before running the script:"
    echo "  export OPENAI_API_KEY=\"your-api-key\""
    exit 1
fi

# --- Determine which Python script to run based on the setting ---
SCRIPT_NAME="scripts/run_${SETTING}.py"

if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: The script for the setting '$SETTING' was not found at '$SCRIPT_NAME'."
    echo "Please ensure the script exists and has the correct name."
    exit 1
fi

# --- Execute the Python Script ---
echo "====================================================="
echo "Starting inference for:"
echo "  Language: $LANGUAGE"
echo "  Task:     $TASK_NUM"
echo "  Setting:  $SETTING"
echo "  Script:   $SCRIPT_NAME"
echo "====================================================="

python "$SCRIPT_NAME" --language "$LANGUAGE" --task "$TASK_NUM"

echo "====================================================="
echo "Script finished for $LANGUAGE, Task $TASK_NUM, Setting $SETTING."
echo "====================================================="

# --- EXAMPLE LOOP TO RUN ALL TASKS FOR A LANGUAGE ---
#
# To run all tasks for English under the zeroshot_no_rationales setting,
# you could use a loop like this:
#
# for task in {1..8}; do
#   ./run_benchmark.sh English $task zeroshot_no_rationales
# done
#
