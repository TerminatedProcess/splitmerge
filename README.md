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

### Option 1: Install from Pre-built Wheel (Recommended)

```bash
# Download or clone this repository
git clone https://github.com/yourusername/splitmerge.git
cd splitmerge

# Install with pipx (recommended for Arch/Garuda Linux)
pipx install dist/splitmerge-1.0.0-py3-none-any.whl

# Use from anywhere
splitmerge /path/to/model_folder
```

**Note for Arch/Garuda Users**: Use `pipx` instead of `pip` due to PEP 668. Install pipx with: `sudo pacman -S python-pipx`

### Option 2: Build and Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/splitmerge.git
cd splitmerge

# Build the wheel
python3 -m build

# Install globally with pipx
pipx install dist/splitmerge-1.0.0-py3-none-any.whl

# Use from anywhere
splitmerge /path/to/model_folder
```

### Option 3: Direct Usage (No Installation)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the script directly
python splitmerge.py /path/to/model_folder
```

## Usage Example

```bash
splitmerge ./qwen3vl # model folder containing model-00001-0f-00004.safetensors, model-00002, model-00003, model00004
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
install        # Install dependencies from requirements.txt
build          # Install package as system command (editable mode)
run            # Quick test run with example model folder

# Usage examples:
source .salias   # Load aliases in current session
install          # Install all dependencies
build            # Make 'splitmerge' available system-wide
run              # Test with qwen3vl folder
```

**Note**: These aliases require Fish shell and the `uv` package manager. If you use a different shell or package manager, you can adapt the commands or use standard pip/virtualenv commands instead.

## Dependencies

- **Python**: 3.8 or higher
- **safetensors**: For reading/writing safetensors files (≥0.4.0)
- **torch**: PyTorch framework (≥2.0.0)
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
