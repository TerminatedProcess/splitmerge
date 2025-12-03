#!/bin/bash
# Install splitmerge for Fish shell

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FISH_FUNCTIONS="$HOME/.config/fish/functions"

echo "Installing splitmerge utility for Fish shell..."

# Create Fish functions directory if it doesn't exist
mkdir -p "$FISH_FUNCTIONS"

# Create Fish function
cat > "$FISH_FUNCTIONS/splitmerge.fish" << EOF
function splitmerge --description 'Merge split safetensors files'
    python3 $SCRIPT_DIR/splitmerge.py \$argv
end
EOF

echo "✓ Created Fish function: $FISH_FUNCTIONS/splitmerge.fish"
echo ""
echo "Installation complete!"
echo "Usage: splitmerge /path/to/model_folder"
echo ""
echo "Note: You may need to install dependencies first:"
echo "  pip install -r requirements.txt"
