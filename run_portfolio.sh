#!/bin/bash

echo "=== Running ROBO UOC ADVISOR - TFG Alberto Garulo ==="

# Make sure we are in the script directory
cd "$(dirname "$0")" || exit 1

# Check Python
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: Python3 is not installed."
    echo "Install it with: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# Create virtual environment if it does not exist
if [ ! -d ".venv" ]; then
    echo "=== Creating virtual environment ==="
    python3 -m venv .venv || {
        echo "ERROR: Failed to create virtual environment."
        echo "On Debian/Ubuntu run:"
        echo "  sudo apt install python3-venv"
        exit 1
    }
fi

# Activate virtual environment
echo "=== Activating virtual environment ==="
source .venv/bin/activate

# Upgrade pip
echo "=== Upgrading pip ==="
python -m pip install --upgrade pip

# Install requirements
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found."
    exit 1
fi

echo "=== Installing requirements ==="
pip install -r requirements.txt

# Run Streamlit app
echo "=== Starting Streamlit application ==="
streamlit run portfolio.py

echo "=== Application finished ==="
