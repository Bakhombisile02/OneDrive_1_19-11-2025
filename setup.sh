#!/bin/bash
set -e

echo "=== Setting up INFO411 Assignment 2 Environment ==="

# 1. Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found. Please install Python 3."
    exit 1
fi

# 2. Create Virtual Environment
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists."
else
    echo "Creating virtual environment in '$VENV_DIR'..."
    python3 -m venv $VENV_DIR
fi

# 3. Activate Virtual Environment
source $VENV_DIR/bin/activate
echo "Virtual environment activated."

# 4. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 5. Install Dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found!"
    exit 1
fi

# 6. Download Data
echo "Downloading dataset..."
python download_data.py

echo "=== Setup Complete! ==="
echo "To start working, run:"
echo "  source .venv/bin/activate"
echo "  python assignment.py"
