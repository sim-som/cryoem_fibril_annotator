# CryoEM Fibril Annotator

A napari-based tool for annotating fibrils in cryo-EM (cryo-electron microscopy) micrographs. This application provides interactive visualization and annotation capabilities for structural biology researchers working with amyloid fibril samples.

## Features

- **Interactive Display**: Real-time visualization of cryo-EM micrographs with napari
- **Power Spectrum Support**: Synchronized display of micrographs and their corresponding power spectra
- **Real-time Filtering**: Butterworth lowpass filtering with Angstrom-based resolution control
- **Flexible Fibril Annotation**: Line and polyline tracing tools for straight **and** curved fibrils
- **Multi-layer Support**: Create separate annotation layers for different fibril types (Aβ42, Tau, α-synuclein, etc.)
- **Memory Efficient**: Handles large datasets (>GB) using Dask arrays and lazy loading
- **Annotation Persistence**: Save and load annotations with full metadata preservation

## Installation

### Using Conda/Mamba (Recommended)

```bash
# Create environment
mamba env create -f environment.yml
# or
conda env create -f environment.yml

# Activate environment
conda activate cryoem-annotator
```

### Automated Setup

```bash
# Full automated setup (recommended)
bash setup_cryoem_env.sh
```

## Usage

### Basic Usage

```bash
# Activate environment first
conda activate cryoem-annotator

# Run on micrograph directory
python cryoem-fibril-annotator.py /path/to/micrographs/
```

### With Power Spectra

```bash
# Run with power spectra
python cryoem-fibril-annotator.py /path/to/micrographs/ --ps_dir /path/to/power_spectra/
```

### Custom File Patterns

```bash
# Custom file patterns
python cryoem-fibril-annotator.py /path/to/micrographs/ --glob_pattern "*.mrc" --ps_dir /path/to/ps/ --ps_glob "*_ctf_2D.mrc"
```


## Annotation System

### Creating Annotations

The application supports multiple annotation layers for different fibril types:

1. **Layer Management**:
   - Use the "Layer Management" widget to create new annotation layers
   - Default suggestions: Aβ42, Tau, α-synuclein
   - Each layer can have a different color for easy identification

2. **Annotation Tools**:
   - **Line Tool** (`L`): For straight fibril segments
   - **Polyline Tool** (`Shift+L`): For multi-segment fibrils with curves
   - **Selection Tool** (`A`): For editing existing annotations
   - **Delete** (`D`): Remove selected annotations

3. **Navigation**:
   - Each micrograph has independent annotations per layer
   - Use the bottom slider or arrow keys to navigate between frames
   - Power spectra automatically synchronize with micrograph navigation

### Saving Annotations

Annotations are saved as NumPy (.npy) files containing comprehensive metadata:

**To save annotations:**
1. Select the desired annotation layer in the napari layer list
2. Use the "Save Annotations" widget
3. Specify the filename (recommended: `{fibril_type}_annotations.npy`)
4. Click "Save Annotations"

**Saved data includes:**
- **Shape coordinates**: All annotation geometries
- **Shape types**: Line, polyline, etc.
- **Layer properties**: Color, name, edge width
- **Pixel size**: For scale consistency
- **File references**: Original MRC filenames for frame mapping
- **Dimensionality**: 2D/3D compatibility information
- **Frame indices**: Which micrograph each annotation belongs to

**Example file structure:**
```
project_directory/
├── micrographs/
├── annotations/
│   ├── Ab42_annotations.npy
│   ├── Tau_annotations.npy
│   └── alpha_synuclein_annotations.npy
└── cryoem-fibril-annotator.py
```

### Loading Annotations

**To load annotations:**
1. Use the "Load Annotations" widget
2. Select the .npy annotation file
3. Click "Load Annotations"

**Loading features:**
- **Automatic layer creation**: Creates a new layer with original name and color
- **Frame mapping**: Automatically reorders annotations if micrograph order changed
- **Validation**: Checks pixel size and dimensionality compatibility
- **Conflict resolution**: Handles duplicate layer names automatically
- **Edge width preservation**: Maintains fibril width based on pixel size

**Compatibility checks:**
- **Pixel size matching**: Warns if pixel sizes differ between sessions
- **File correspondence**: Maps annotations to correct micrographs even if file order changed

### Converting Annotations to RELION Format

The `convert_to_star.py` script converts saved annotations (.npy files) to RELION-compatible .star files for downstream cryo-EM processing. The script handles coordinate system conversion between napari (top-left origin) and RELION (bottom-left origin).

#### Basic Conversion

**Generate particle coordinates along fibrils:**
```bash
# Recommended: Provide micrograph dimensions for fast coordinate conversion
python convert_to_star.py annotations.npy --mic_shape 4096 4096

# Common camera formats:
python convert_to_star.py annotations.npy --mic_shape 4096 4096   # Falcon 4
python convert_to_star.py annotations.npy --mic_shape 4092 5760   # Gatan K3 (non-superresolution)

# Custom inter-box spacing (default: 100 Å)
python convert_to_star.py annotations.npy --inter_box_distance 150 --mic_shape 4096 4096

# Custom output filename
python convert_to_star.py annotations.npy -o fibrils.star --mic_shape 4096 4096
```

#### Manual Pick Files (Start-End Coordinates)

**Export filament endpoints for RELION helical processing:**
```bash
# Export per-micrograph manual pick files
python convert_to_star.py annotations.npy --manualpick --mic_shape 4096 4096

# Split multi-point paths into individual segments
python convert_to_star.py annotations.npy --manualpick --split_paths --mic_shape 4096 4096

# Custom output directory
python convert_to_star.py annotations.npy --manualpick --manualpick_dir my_picks/ --mic_shape 4096 4096
```

#### Alternative: Auto-detect Micrograph Dimensions

**Load MRC files to determine dimensions (slower but accurate for mixed sizes):**
```bash
python convert_to_star.py annotations.npy --manualpick --mrc_dir /path/to/micrographs/
```

#### Conversion Modes

**1. Particle Coordinates Mode (default)**
- Generates evenly-spaced particle coordinates along each fibril
- Includes helical tube ID and track length for RELION
- Calculates psi angle (filament orientation)
- Suitable for helical reconstruction workflows

**Output format:**
```
_rlnCoordinateX #1
_rlnCoordinateY #2
_rlnMicrographName #3
_rlnAnglePsi #4
_rlnHelicalTrackLength #5
_rlnHelicalTubeID #6
```

**2. Manual Pick Mode (`--manualpick`)**
- Exports start-end coordinates for each fibril
- Creates one .star file per micrograph
- Compatible with RELION manual picking format
- Suitable for helical picking workflows

**Output format:**
```
_rlnCoordinateX #1
_rlnCoordinateY #2
_rlnParticleSelectionType #3
_rlnAnglePsi #4
_rlnAutopickFigureOfMerit #5
```

#### Key Options

- `--mic_shape HEIGHT WIDTH`: Micrograph dimensions (recommended for speed)
- `--mrc_dir PATH`: Load MRC files to auto-detect dimensions (slower)
- `--manualpick`: Export start-end coordinates instead of particle positions
- `--split_paths`: Split polylines into individual segments
- `--inter_box_distance`: Spacing between particles in Angstroms (default: 100)
- `--box_size`: Box size for particle extraction in pixels (default: 256)
- `-o, --output`: Custom output filename

#### Important Notes

⚠️ **Coordinate System Conversion**: Either `--mic_shape` or `--mrc_dir` is **required** to correctly convert y-coordinates from napari's top-left origin to RELION's bottom-left origin. Without this, coordinates will appear vertically mirrored in RELION.

✅ **Recommendation**: Use `--mic_shape` when all micrographs have the same dimensions (much faster than loading MRC files).

#### Example Workflow RELION:

1. Annotate fibrils in napari
```bash
python cryoem-fibril-annotator.py /path/to/micrographs/
```

2. Save annotations as Ab42_annotations.npy

3. Convert to RELION manual pick files
```bash
python convert_to_star.py Ab42_annotations.npy --manualpick --mic_shape 4096 4096
```

4. Output: Ab42_annotations_manualpick/ directory with per-micrograph .star files
- micrograph_001_manualpick.star
- micrograph_002_manualpick.star
- ...

5. Import to RELION for helical processing


## Interactive Controls

### Display Controls
- **Power Spectrum Toggle**: Checkbox to show/hide power spectra
- **PS Scale**: Adjust power spectrum display size (0.1x - 5.0x)
- **Synchronized Navigation**: Power spectra automatically sync with micrograph navigation

### Filtering Controls
- **Resolution Threshold**: Real-time Butterworth lowpass filtering in Angstroms
- **Filter Order**: Control filter sharpness (1-10, default: 4)
- **Preview Mode**: Toggle between original and filtered views

### Annotation Controls
- **Layer Creation**: Create new layers for different fibril types
- **Color Selection**: Choose from 8 predefined colors per layer
- **Save/Load**: Persistent annotation storage with full metadata

## Technical Details

### Architecture
- **Main Application**: `cryoem-fibril-annotator.py` - Primary interactive annotation tool
- **File Utility**: `cryoem-file-finder.py` - Helper for finding corresponding cryo-EM files
- **Core Class**: `CryoEMFibrilAnnotator` - Main workflow handler

### Key Dependencies
- **napari** (≥0.4.18): Visualization framework
- **dask** (≥2023.5): Lazy loading and parallel processing
- **mrcfile** (≥1.4): MRC file format support
- **scikit-image** (≥0.20): Image processing and filtering
- **magicgui** (≥0.7): Interactive GUI controls
- **numpy**: Annotation data storage and manipulation

### Memory Management
- **Lazy Loading**: Dask arrays for memory-efficient handling of large datasets
- **Chunked Processing**: Prevents memory overflow with large files
- **Delayed Evaluation**: Filter operations computed on-demand

### File Format Support
- **MRC Files**: Primary format with robust error handling
- **Stack Support**: Both individual files and stacks supported
- **Annotation Files**: NumPy format (.npy) with pickle support for complex metadata

### Annotation Data Format

The annotation file format preserves all necessary information for reproducible analysis:

```python
annotations = {
    'shapes': [array1, array2, ...],        # Coordinate arrays for each annotation
    'shape_types': ['line', 'polyline'],     # Shape type for each annotation
    'properties': {...},                     # Additional properties
    'pixel_size': 1.23,                      # Pixel size in Angstroms
    'mrc_files': ['file1.mrc', 'file2.mrc'], # Original filenames
    'ndim': 3,                               # 2D or 3D annotations
    'layer_name': 'Aβ42',                    # Layer identifier
    'edge_color': 'red',                     # Visualization color
    'frame_indices': [0, 1, 0, 2, ...]      # Frame assignment for each annotation
}
```

## Scientific Context

This tool is designed for cryo-EM structural biology workflows, specifically:
- Manual annotation of amyloid fibril structures in micrographs
- Ground truth generation for machine learning training datasets
- Multi-class fibril annotation for comparative studies

## Keyboard Shortcuts

- **L**: Line tool (straight segments)
- **Shift+L**: Polyline tool (multi-segment)
- **A**: Selection/edit tool
- **D**: Delete selected annotations
- **M**: Pan/zoom mode
- **Escape**: Cancel current annotation
- **Arrow keys**: Navigate between micrographs
- **Mouse wheel**: Zoom in/out
- **Click+drag**: Pan (in pan mode)

## Best Practices

### Annotation Workflow
1. **Layer Organization**: Create separate layers for each fibril type
2. **Consistent Naming**: Use descriptive layer names (e.g., "Aβ42_fibrils", "Tau_tangles")
3. **Regular Saving**: Save annotations frequently to prevent data loss
4. **Cross beta signal**: Use power spectra to verify fibril presence and orientation
5. **Resolution Filtering**: Apply appropriate lowpass filtering to reduce noise


## Development Notes

- This is a standalone application, not a formal napari plugin
- No formal testing framework - manual verification recommended
- Interactive testing through the setup script
- Handles float16/float32 conversions automatically
- GUI requires display environment (not suitable for headless servers)

## License

[Add license information]
