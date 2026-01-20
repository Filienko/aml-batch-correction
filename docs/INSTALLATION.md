# Installation Guide

## Quick Install

```bash
# Clone the repository
git clone https://github.com/Filienko/aml-batch-correction.git
cd aml-batch-correction

# Install with pip
pip install -e .
```

## Requirements

### Python Version
- Python >= 3.8

### Core Dependencies
These are installed automatically:
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- anndata >= 0.8.0
- scanpy >= 1.9.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- pyyaml >= 5.4.0

### Optional Dependencies

#### For Foundation Models

**SCimilarity**:
```bash
pip install scimilarity
```

**scVI**:
```bash
pip install scvi-tools
```

#### All Optional Dependencies
```bash
pip install -e ".[all]"
```

## Installation Methods

### Method 1: Development Install (Recommended)

For development or if you want to modify the code:

```bash
git clone https://github.com/Filienko/aml-batch-correction.git
cd aml-batch-correction
pip install -e .
```

Benefits:
- Changes to code are immediately reflected
- Easy to contribute back
- Can run examples directly

### Method 2: Standard Install

```bash
git clone https://github.com/Filienko/aml-batch-correction.git
cd aml-batch-correction
pip install .
```

### Method 3: From Requirements File

```bash
pip install -r requirements.txt
```

Then use the package without installing (add to Python path).

## Verify Installation

```bash
# Check CLI is available
sccl --help

# Check version
python -c "import sccl; print(sccl.__version__)"

# Run a quick test
python examples/generate_synthetic_data.py
sccl evaluate --data data/synthetic_example.h5ad --model random_forest --target cell_type
```

## Platform-Specific Notes

### Linux
Should work out of the box. If you encounter issues:

```bash
# Install build dependencies
sudo apt-get install build-essential python3-dev

# Then retry installation
pip install -e .
```

### macOS
Should work with system Python or homebrew Python:

```bash
# If using homebrew Python
brew install python@3.10

# Then install
pip3 install -e .
```

### Windows
Should work with Anaconda or standard Python:

```bash
# Recommended: use Anaconda
conda create -n sccl python=3.10
conda activate sccl
pip install -e .
```

## Using Conda (Alternative)

If you prefer conda:

```bash
# Create environment
conda create -n sccl python=3.10
conda activate sccl

# Install dependencies
conda install numpy pandas scipy scikit-learn matplotlib seaborn
conda install -c conda-forge scanpy python-anndata

# Install SCCL
pip install -e .
```

## Docker (For Reproducibility)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone repository
RUN git clone https://github.com/Filienko/aml-batch-correction.git

# Install
WORKDIR /app/aml-batch-correction
RUN pip install -e .

# Optional: install all models
RUN pip install -e ".[all]"

CMD ["bash"]
```

Build and run:

```bash
docker build -t sccl .
docker run -it -v $(pwd)/data:/app/data sccl
```

## Google Colab

To use in Google Colab:

```python
# In a Colab notebook
!git clone https://github.com/Filienko/aml-batch-correction.git
%cd aml-batch-correction
!pip install -e .

# Now you can use it
from sccl import Pipeline
```

## Troubleshooting

### "Command not found: sccl"

The CLI might not be in your PATH. Try:

```bash
python -m sccl.cli.main --help
```

Or reinstall with:
```bash
pip uninstall sccl
pip install -e .
```

### "ImportError: No module named sccl"

Make sure you're in the right environment:

```bash
which python
pip list | grep sccl
```

If not installed, run `pip install -e .` again.

### "ModuleNotFoundError: No module named 'scanpy'"

Core dependencies not installed:

```bash
pip install -r requirements.txt
```

### "SCimilarity not installed"

This is optional. Install if needed:

```bash
pip install scimilarity
```

### "scvi-tools not installed"

This is optional. Install if needed:

```bash
pip install scvi-tools
```

### Permission Errors

On Linux/Mac, you might need:

```bash
pip install --user -e .
```

Or use a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
pip install -e .
```

## Virtual Environment (Recommended)

Always use a virtual environment to avoid conflicts:

### Using venv

```bash
# Create environment
python -m venv sccl-env

# Activate
source sccl-env/bin/activate  # Linux/Mac
# or: sccl-env\Scripts\activate  # Windows

# Install
pip install -e .
```

### Using conda

```bash
# Create environment
conda create -n sccl python=3.10

# Activate
conda activate sccl

# Install
pip install -e .
```

## Updating

To get the latest version:

```bash
cd aml-batch-correction
git pull
pip install -e . --upgrade
```

## Uninstalling

```bash
pip uninstall sccl
```

## Development Setup

For contributing to SCCL:

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/aml-batch-correction.git
cd aml-batch-correction

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

## Next Steps

After installation:

1. **Verify**: Run `sccl --help`
2. **Generate test data**: `python examples/generate_synthetic_data.py`
3. **Run quick test**: `sccl evaluate --data data/synthetic_example.h5ad --model random_forest --target cell_type`
4. **Read guides**:
   - [Quick Start](QUICKSTART.md)
   - [User Guide](USER_GUIDE.md)
   - [Model Guide](MODELS.md)

## Getting Help

If you encounter issues:

1. Check this guide
2. Search existing issues: https://github.com/Filienko/aml-batch-correction/issues
3. Create a new issue with:
   - Your OS and Python version
   - Full error message
   - Steps to reproduce
