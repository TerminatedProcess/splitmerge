#!/usr/bin/env python3
"""
Merge split safetensors files into a single file.

Standalone utility - only depends on safetensors library.

Usage:
    python splitmerge.py /path/to/model_folder

Example:
    python splitmerge.py ./qwen3vl
"""

import sys
import shutil
from pathlib import Path
import re

# Direct safetensors imports - no other dependencies needed
from safetensors import safe_open
from safetensors.torch import save_file


def is_lfs_pointer(file_path: Path) -> bool:
    """
    Check if a file is a Git LFS pointer instead of actual data.

    LFS pointers are small text files that look like:
    version https://git-lfs.github.com/spec/v1
    oid sha256:...
    size ...
    """
    # LFS pointers are always very small (< 200 bytes)
    if file_path.stat().st_size > 200:
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            return 'git-lfs.github.com' in first_line
    except (UnicodeDecodeError, IOError):
        # If we can't read it as text, it's not an LFS pointer
        return False


def merge_safetensor_files(shard_files, output_file):
    """
    Merge multiple safetensors files into a single file.

    Borrowed from reshard-safetensors project.

    Args:
        shard_files: List of paths to safetensors files to merge
        output_file: Path for output merged file
    """
    tensors = {}
    metadata = None

    for file in shard_files:
        with safe_open(file, framework="pt") as sf_tsr:
            # Get metadata from first file
            if metadata is None:
                metadata = sf_tsr.metadata()

            # Load all tensors from this shard
            for layer in sf_tsr.keys():
                tensor = sf_tsr.get_tensor(str(layer))
                tensors[str(layer)] = tensor

    # Save merged tensors
    save_file(tensors, output_file, metadata)


def get_split_files(folder_path: Path):
    """
    Find all model-*-of-*.safetensors files in folder.

    Returns:
        tuple: (shard_files, total_expected) or (None, None) if invalid
    """
    # Pattern: model-00001-of-00004.safetensors
    pattern = re.compile(r'model-(\d+)-of-(\d+)\.safetensors')

    shard_files = []
    total_expected = None

    for file_path in folder_path.glob('*.safetensors'):
        match = pattern.match(file_path.name)
        if match:
            shard_num = int(match.group(1))
            total_shards = int(match.group(2))

            if total_expected is None:
                total_expected = total_shards
            elif total_expected != total_shards:
                print(f"❌ Error: Inconsistent shard counts found")
                print(f"   Expected: {total_expected}, Found: {total_shards} in {file_path.name}")
                return None, None

            shard_files.append((shard_num, file_path))

    if not shard_files:
        return None, None

    # Sort by shard number
    shard_files.sort(key=lambda x: x[0])

    return [f[1] for f in shard_files], total_expected


def validate_shards(shard_files, total_expected):
    """
    Validate that all shards are present and not LFS pointers.

    Returns:
        tuple: (success, error_message)
    """
    # Check count
    if len(shard_files) != total_expected:
        return False, f"Missing shards: found {len(shard_files)}, expected {total_expected}"

    # Check sequential numbering
    for i, file_path in enumerate(shard_files, start=1):
        # Extract shard number from filename
        match = re.search(r'model-(\d+)-of-\d+\.safetensors', file_path.name)
        if match:
            shard_num = int(match.group(1))
            if shard_num != i:
                return False, f"Non-sequential shard numbering: expected {i:05d}, found {shard_num:05d}"

    # Check for LFS pointers
    for file_path in shard_files:
        if is_lfs_pointer(file_path):
            return False, f"LFS pointer detected (not downloaded): {file_path.name}"

    return True, None


def merge_model_shards(folder_path: str):
    """
    Main function to merge split safetensors files.

    Args:
        folder_path: Path to folder containing split files
    """
    folder = Path(folder_path).resolve()

    if not folder.exists():
        print(f"❌ Error: Folder not found: {folder}")
        return False

    if not folder.is_dir():
        print(f"❌ Error: Not a directory: {folder}")
        return False

    print(f"📁 Processing folder: {folder.name}")
    print(f"   Full path: {folder}")

    # Find split files
    shard_files, total_expected = get_split_files(folder)

    if not shard_files:
        print(f"❌ Error: No model-*-of-*.safetensors files found in {folder}")
        return False

    print(f"✓ Found {len(shard_files)} shard files (expected: {total_expected})")

    # Validate shards
    valid, error_msg = validate_shards(shard_files, total_expected)
    if not valid:
        print(f"❌ Validation failed: {error_msg}")
        return False

    print("✓ All shards validated (present and not LFS pointers)")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in shard_files)
    total_size_gb = total_size / (1024**3)
    print(f"✓ Total size: {total_size_gb:.2f} GB")

    # Create merged subfolder
    merged_folder = folder / "merged"
    if merged_folder.exists():
        print(f"🗑️  Removing existing merged folder...")
        shutil.rmtree(merged_folder)

    merged_folder.mkdir()
    print(f"✓ Created merged folder: {merged_folder}")

    # Output filename from folder name
    output_filename = f"{folder.name}.safetensors"
    output_path = merged_folder / output_filename

    print(f"🔄 Merging shards into: {output_filename}")
    print(f"   This may take a while for large models...")

    # Merge the files
    try:
        merge_safetensor_files(
            [str(f) for f in shard_files],
            output_file=str(output_path)
        )
    except Exception as e:
        print(f"❌ Merge failed: {e}")
        return False

    # Verify output
    if not output_path.exists():
        print(f"❌ Error: Output file not created")
        return False

    output_size = output_path.stat().st_size
    output_size_gb = output_size / (1024**3)

    print(f"✅ Merge complete!")
    print(f"   Output: {output_path}")
    print(f"   Size: {output_size_gb:.2f} GB")

    # Verify size matches (approximately - some overhead is expected)
    size_diff_percent = abs(output_size - total_size) / total_size * 100
    if size_diff_percent > 5:
        print(f"⚠️  Warning: Size difference is {size_diff_percent:.1f}% (expected ~0-5%)")
    else:
        print(f"✓ Size verification passed (diff: {size_diff_percent:.1f}%)")

    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: python merge_shards.py /path/to/model_folder")
        print("")
        print("Example:")
        print("  python merge_shards.py ./qwen3vl")
        print("")
        print("This will:")
        print("  1. Find all model-*-of-*.safetensors files")
        print("  2. Validate all shards are present (not LFS pointers)")
        print("  3. Merge into a single file: qwen3vl.safetensors")
        print("  4. Save in subfolder: qwen3vl/merged/")
        sys.exit(1)

    folder_path = sys.argv[1]
    success = merge_model_shards(folder_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
