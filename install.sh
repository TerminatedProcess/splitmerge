#!/bin/bash
# Install splitmerge utility system-wide

echo "Installing splitmerge utility..."

# Install package
pip install -e .

# Create alias in your shell config
SHELL_CONFIG=""
if [ -f "$HOME/.bashrc" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
fi

if [ -n "$SHELL_CONFIG" ]; then
    # Check if alias already exists
    if ! grep -q "alias splitmerge=" "$SHELL_CONFIG"; then
        echo "" >> "$SHELL_CONFIG"
        echo "# Splitmerge utility" >> "$SHELL_CONFIG"
        echo "alias splitmerge='python $(pwd)/splitmerge.py'" >> "$SHELL_CONFIG"
        echo "✓ Added alias to $SHELL_CONFIG"
        echo "  Run: source $SHELL_CONFIG (or restart terminal)"
    else
        echo "✓ Alias already exists in $SHELL_CONFIG"
    fi
fi

echo ""
echo "Installation complete!"
echo "Usage: splitmerge /path/to/model_folder"
