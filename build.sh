#!/bin/bash
# Install system deps
apt-get update && apt-get install -y $(cat apt-packages.txt)

# Special numpy install first
python -m pip install --upgrade "numpy==1.24.4" --only-binary numpy

# Install remaining requirements
python -m pip install -r requirements.txt --no-build-isolation
