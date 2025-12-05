# Author: Mark Ryan
# Date: 12/3/2025
#
# Split Merge - Safetensors Shard files Merger

Standalone utility to merge split safetensors files into a single file.

## Purpose

Large AI models from HuggingFace are often distributed as multiple shard files:
- `model-00001-of-00004.safetensors`
- `model-00002-of-00004.safetensors`
- `model-00003-of-00004.safetensors`
- `model-00004-of-00004.safetensors`

This tool merges them into a single `.safetensors` file for easier use with tools like ComfyUI and InvokeAI.

## Quick Start

### Prerequisites

- **Python 3.12.10** (tested and recommended - Python 3.13+ NOT supported due to PyTorch compatibility)
- **pipx** for installation: `sudo pacman -S python-pipx` (Arch/Garuda)
- **uv** package manager for building: Install from [astral.sh/uv](https://astral.sh/uv)

### Option 1: Build and Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/splitmerge.git
cd splitmerge

# Create virtual environment with Python 3.12.10 using uv
# Review file .salias to make it easy to quickly install.
mkuv .venv 3.12.10
source .venv/bin/activate  # or use direnv

# Install build dependencies
uv pip install -r requirements.txt
uv pip install build

# Build the wheel
python -m build

# Install globally with pipx (using the .venv Python 3.12.10)
pipx install --python .venv/bin/python dist/splitmerge-1.0.0-py3-none-any.whl

# Use from anywhere
splitmerge /path/to/model_folder
```

**For Fish shell users**: Use the included `.salias` shortcuts:
```fish
source .salias
mkenv          # Create .venv with Python 3.12.10
installenv     # Install dependencies
build          # Build wheel and install with pipx
```

### Option 2: Direct Usage (No Installation)

```bash
# Create virtual environment with Python 3.12.10
# Note: Requires Python 3.12.10 to be installed via uv or system package manager
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the script directly
python splitmerge.py /path/to/model_folder
```

**Important**: This project is tested with Python 3.12.10 only. Python 3.13+ will cause segmentation faults due to PyTorch incompatibility.

## Usage Example

```bash
splitmerge ./qwen3vl # model folder containing model-00001-of-00004.safetensors, model-00002, model-00003, model-00004
```
## Features

- **LFS Detection**: Automatically detects Git LFS pointers (undownloaded files)
- **Validation**: Ensures all shards are present and sequential
- **Smart Naming**: Uses folder name for output file
- **Clean Output**: Destroys and re-creates fresh `merged/` subfolder each run utility is re-ran.
- **Progress Info**: Shows total size and validation status
- **Size Verification**: Confirms merged file matches expected size

## Output

The merged file will be created at:
```
{folder_name}/merged/{folder_name}.safetensors
```

You can then use this merged file with your AI tools or import it into model management systems.

## Development Environment

This project uses Fish shell with custom development tools for enhanced productivity.

### `.envrc` - Automatic Environment Activation

The `.envrc` file enables automatic virtual environment activation using [direnv](https://direnv.net/):

```bash
# When you cd into this directory, direnv automatically:
# - Activates the .venv virtual environment
# - Deactivates when you leave the directory
```

**Setup direnv** (one-time):
```bash
# Install direnv
sudo pacman -S direnv  # Arch/Garuda
# or: brew install direnv  # macOS

# Add to your shell config (~/.config/fish/config.fish for Fish)
direnv hook fish | source

# Allow this directory
direnv allow
```

### `.salias` - Development Shortcuts

The `.salias` file contains Fish shell aliases and functions for common development tasks:

```fish
# Environment Management
mkenv          # Create new virtual environment (.venv with Python 3.12.10)
rmenv          # Remove virtual environment and exit

# Development
installenv     # Install dependencies from requirements.txt
buildwheel     # Build wheel package
installwheel   # Install wheel with pipx
build          # Build and install (buildwheel + installwheel)
run            # Quick test run with example model folder

# Usage examples:
source .salias   # Load aliases in current session
mkenv            # Create .venv with Python 3.12.10
installenv       # Install all dependencies
build            # Build wheel and install globally with pipx
run              # Test with qwen3vl folder
```

**Note**: These aliases require Fish shell and the `uv` package manager. If you use a different shell or package manager, you can adapt the commands or use standard pip/virtualenv commands instead.

## Dependencies

- **Python**: 3.12.10 (tested - Python 3.13+ NOT supported)
- **safetensors**: For merging model-#-of-# sharded safetensors files (≥0.4.0)
- **torch**: PyTorch framework for tensor operations (≥2.0.0)
- **packaging**: Python packaging utilities (≥21.0)
- **numpy**: Numerical computing library (≥1.20.0)

## Project Structure

```
splitmerge/
├── splitmerge.py       # Main utility script
├── setup.py            # Package installation configuration
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── LICENSE             # MIT License
├── .gitignore          # Git ignore rules
├── .envrc              # Direnv configuration (optional)
└── .salias             # Fish shell aliases (optional)
```

## How It Works

1. **Detection**: Scans folder for files matching `model-*-of-*.safetensors` pattern
2. **Validation**:
   - Verifies all expected shards are present
   - Checks for sequential numbering (00001, 00002, ...)
   - Detects Git LFS pointers (files that haven't been downloaded)
3. **Merging**:
   - Loads tensors from all shards
   - Preserves metadata from first shard
   - Combines into single safetensors file
4. **Output**:
   - Saves to `{folder}/merged/{folder_name}.safetensors`
   - Verifies output size matches input (within 5% tolerance)

## Troubleshooting

**Error: "Package 'splitmerge' requires a different Python: 3.13.7 not in '<3.13,>=3.8'"**
- You're using Python 3.13+ which is not supported
- Install Python 3.12.10 using `uv`: `uv python install 3.12.10`
- Use pipx with Python 3.12: `pipx install --python /path/to/python3.12 <wheel>`

**Segmentation fault when running splitmerge**
- This is caused by PyTorch incompatibility with Python 3.13+
- Uninstall: `pipx uninstall splitmerge`
- Reinstall with Python 3.12.10 (see Quick Start)

**Error: "LFS pointer detected (not downloaded)"**
- Your model files are Git LFS pointers, not actual data
- Run `git lfs pull` in the model directory to download actual files

**Error: "Missing shards"**
- Not all expected shard files are present
- Check the model download completed successfully

**Error: "Non-sequential shard numbering"**
- Shard files are missing or improperly named
- Ensure you have a complete set: 00001, 00002, 00003, etc.

## Credits

Merge logic adapted from [reshard-safetensors](https://github.com/NotTheStallion/reshard-safetensors) by NotTheStallion.

## License

MIT License - See [LICENSE](LICENSE) file for details.
