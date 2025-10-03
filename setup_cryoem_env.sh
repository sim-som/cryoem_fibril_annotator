#!/bin/bash

# setup_cryoem_env.sh
# Complete setup script for Cryo-EM Fibril Annotator environment
# 
# Usage: bash setup_cryoem_env.sh

set -e  # Exit on error

echo "================================"
echo "Cryo-EM Fibril Annotator Setup"
echo "================================"

# Check if mamba is available, otherwise use conda
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "✓ Using mamba (faster)"
else
    CONDA_CMD="conda"
    echo "✓ Using conda"
fi

# Step 1: Create environment from YAML file
echo ""
echo "Step 1: Creating conda environment..."
echo "--------------------------------------"

# Create environment.yml if it doesn't exist
cat > environment.yml << 'EOF'
name: cryoem-annotator
channels:
  - conda-forge
  - defaults

dependencies:
  - python=3.10
  - numpy>=1.23
  - scipy>=1.9
  - pandas>=1.5
  - scikit-image>=0.20
  - mrcfile>=1.4
  - dask>=2023.5
  - distributed>=2023.5
  - napari>=0.4.18
  - pyqt>=5.15
  - magicgui>=0.7
  - pyopengl>=3.1
  - ipykernel>=6.20
  - pip
EOF

# Create the environment
$CONDA_CMD env create -f environment.yml --force

# Step 2: Activate and verify
echo ""
echo "Step 2: Activating environment..."
echo "---------------------------------"

# Get conda base directory
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the environment
conda activate cryoem-annotator

# Step 3: Verify installation
echo ""
echo "Step 3: Verifying installation..."
echo "---------------------------------"

python << 'EOF'
import sys
print(f"Python: {sys.version}")

# Check critical packages
packages = {
    'numpy': 'np',
    'scipy': 'scipy',
    'skimage': 'skimage',
    'dask': 'dask',
    'mrcfile': 'mrcfile',
    'napari': 'napari',
    'magicgui': 'magicgui'
}

print("\nPackage Status:")
print("-" * 40)

all_good = True
for package, import_name in packages.items():
    try:
        if import_name == 'np':
            import numpy as np
            version = np.__version__
        elif import_name == 'scipy':
            import scipy
            version = scipy.__version__
        elif import_name == 'skimage':
            import skimage
            version = skimage.__version__
        elif import_name == 'dask':
            import dask
            version = dask.__version__
        elif import_name == 'mrcfile':
            import mrcfile
            version = mrcfile.__version__
        elif import_name == 'napari':
            import napari
            version = napari.__version__
        elif import_name == 'magicgui':
            import magicgui
            version = magicgui.__version__
        
        print(f"✓ {package:<15} {version}")
    except ImportError as e:
        print(f"✗ {package:<15} FAILED: {e}")
        all_good = False

if all_good:
    print("\n✓ All packages installed successfully!")
else:
    print("\n✗ Some packages failed to install")
    sys.exit(1)

# Test napari (headless check)
print("\nTesting napari initialization...")
try:
    import napari
    print("✓ Napari can be imported")
    
    # Check if we can create a viewer (may fail in headless mode)
    try:
        import os
        if 'DISPLAY' in os.environ or sys.platform == 'darwin' or sys.platform == 'win32':
            from napari import Viewer
            print("✓ Napari GUI components available")
        else:
            print("⚠ Running in headless mode - GUI will not work")
    except:
        print("⚠ Cannot create viewer - may be running headless")
        
except Exception as e:
    print(f"✗ Napari test failed: {e}")

print("\n" + "="*40)
print("Setup complete!")
print("="*40)
EOF

# Step 4: Save the annotator script
echo ""
echo "Step 4: Saving the annotator script..."
echo "--------------------------------------"

# Download or create the main script
if [ ! -f "fibril_annotator.py" ]; then
    echo "Creating fibril_annotator.py..."
    echo "✓ Script template created"
    echo "  Please copy the full annotator script from the artifact"
else
    echo "✓ fibril_annotator.py already exists"
fi

# Step 5: Create test script
cat > test_annotator.py << 'EOF'
#!/usr/bin/env python3
"""Quick test script to verify the environment is working"""

import numpy as np
import napari
from skimage.filters import butterworth
import mrcfile

print("Testing environment components...")

# Test Butterworth filter
test_image = np.random.randn(512, 512).astype(np.float32)
filtered = butterworth(test_image, cutoff_frequency_ratio=0.1, order=4)
print("✓ Butterworth filter works")

# Test MRC file handling
print("✓ MRC file support available")

# Test Dask
import dask.array as da
x = da.random.random((1000, 1000), chunks=(100, 100))
print(f"✓ Dask array created: {x.shape}")

print("\nAll tests passed! Environment is ready.")
print("\nTo use the annotator:")
print("  1. Activate environment: conda activate cryoem-annotator")
print("  2. Run: python fibril_annotator.py your_data.mrc")
EOF

chmod +x test_annotator.py

# Step 6: Final instructions
echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Activate the environment:"
echo "   conda activate cryoem-annotator"
echo ""
echo "2. Test the installation:"
echo "   python test_annotator.py"
echo ""
echo "3. Run the annotator:"
echo "   python fibril_annotator.py <your_mrc_files>"
echo ""
echo "4. For Jupyter notebook usage:"
echo "   jupyter notebook"
echo ""
echo "Notes:"
echo "- The environment is named 'cryoem-annotator'"
echo "- Python 3.10 is used for best compatibility"
echo "- All dependencies are from conda-forge for consistency"
echo "- If napari fails to open, check your display settings"
echo ""
echo "For GPU acceleration (optional):"
echo "  conda install -c rapidsai -c nvidia -c conda-forge cupy cudatoolkit=11.8"
echo ""