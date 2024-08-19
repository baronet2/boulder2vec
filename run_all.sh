#!/bin/bash

run_notebook() {
    jupyter nbconvert --to notebook --execute $1
}

# Check if MPS is available and set USE_MPS environment variable
if python -c "import torch; print(torch.backends.mps.is_available())" | grep -q "True"; then
    export USE_MPS=1
    echo "MPS is available. Using MPS for PyTorch."
else
    unset USE_MPS
    echo "MPS is not available. Using CPU for PyTorch."
fi

# Run pmf.py
echo "Running pmf.py..."
python pmf.py

# Run lr.py
echo "Running lr.py..."
python lr.py

# Run eval.ipynb
echo "Running eval.ipynb..."
run_notebook eval.ipynb

echo "All tasks completed."