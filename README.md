# Split Merge - Safetensors Shard Merger

Standalone utility to merge split safetensors files into a single file.

## Purpose

Large AI models from HuggingFace are often distributed as multiple shard files:
- `model-00001-of-00004.safetensors`
- `model-00002-of-00004.safetensors`
- `model-00003-of-00004.safetensors`
- `model-00004-of-00004.safetensors`

This tool merges them into a single `.safetensors` file for easier use with tools like ComfyUI and InvokeAI.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python splitmerge.py /path/to/model_folder
```

### Example

```bash
python splitmerge.py ./qwen3vl
```

This will:
1. ✓ Find all `model-*-of-*.safetensors` files in the folder
2. ✓ Validate all shards are present (not Git LFS pointers)
3. ✓ Merge into a single file named `qwen3vl.safetensors`
4. ✓ Save in subfolder `qwen3vl/merged/`

## Features

- **LFS Detection**: Automatically detects Git LFS pointers (undownloaded files)
- **Validation**: Ensures all shards are present and sequential
- **Smart Naming**: Uses folder name for output file
- **Clean Output**: Creates fresh `merged/` subfolder each run
- **Progress Info**: Shows total size and validation status

## Output

The merged file will be created at:
```
{folder_name}/merged/{folder_name}.safetensors
```

You can then use this merged file with your AI tools or import it into model management systems.

## Dependencies

- `safetensors` - For reading/writing safetensors files
- `torch` - PyTorch (required by safetensors)

## License

MIT
