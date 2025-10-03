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