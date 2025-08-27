# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a napari-based tool for annotating fibrils in cryo-EM (cryo-electron microscopy) micrographs. The application provides interactive visualization and annotation capabilities for structural biology researchers working with amyloid fibril samples.

## Environment Setup

The project uses conda/mamba for environment management:

```bash
# Create environment
mamba env create -f environment.yml
# or
conda env create -f environment.yml

# Activate environment
conda activate cryoem-annotator

# Full automated setup (recommended)
bash setup_cryoem_env.sh
```

## Running the Application

```bash
# Activate environment first
conda activate cryoem-annotator

# Run on micrograph directory
python cryoem-fibril-annotator.py /path/to/micrographs/

# Run with power spectra
python cryoem-fibril-annotator.py /path/to/micrographs/ --ps_dir /path/to/power_spectra/

# Custom file patterns
python cryoem-fibril-annotator.py /path/to/micrographs/ --glob_pattern "*.mrc" --ps_dir /path/to/ps/ --ps_glob "*_ctf_2D.mrc"

# Find corresponding files (utility)
python cryoem-file-finder.py
```

## Core Architecture

### Main Components

- **`cryoem-fibril-annotator.py`**: Primary application providing interactive fibril annotation
- **`cryoem-file-finder.py`**: Utility for finding corresponding cryo-EM files (micrographs and power spectra)
- **`CryoEMFibrilAnnotator`**: Main class handling the complete workflow

### Key Design Patterns

- **Lazy Loading**: Uses Dask arrays for memory-efficient handling of large MRC file stacks
- **Interactive Filtering**: Real-time Butterworth lowpass filtering with Angstrom-based resolution control
- **Robust Error Handling**: MRC file corruption handled with permissive mode fallbacks
- **Modular Architecture**: Clear separation between data loading, processing, and visualization

### Memory Management

The application handles large cryo-EM datasets (often >GB) through:
- Dask arrays for delayed loading
- Chunked processing to avoid memory overflow
- Lazy evaluation of filter operations

### File Format Support

- **MRC files**: Primary format with robust corruption handling
- **Stacks and individual files**: Both supported with per-frame annotation tracking
- **Permissive mode fallback**: Handles corrupted/non-standard MRC files

## Key Dependencies

- **napari**: Visualization framework (>=0.4.18)
- **dask**: Lazy loading and parallel processing (>=2023.5)
- **mrcfile**: MRC file format support (>=1.4)
- **scikit-image**: Butterworth filtering (>=0.20)
- **magicgui**: Interactive GUI controls (>=0.7)

## Interactive Features

### Display Controls
- Toggle power spectrum visibility with checkbox control
- Power spectra automatically synchronized with micrograph navigation
- Separate colormap (viridis) for power spectra distinction

### Filtering Controls
- Resolution threshold in Angstroms (real-time adjustment)
- Filter order control (Butterworth lowpass)
- Interactive preview with original/filtered view toggle

### Annotation Tools
- Line and polyline tracing for fibril annotation
- Per-frame annotation in stack mode
- Save annotations (load functionality is TODO)

## Development Notes

### No Formal Testing
- No unit test framework
- Manual verification through setup script
- Interactive testing recommended

### File Structure
This is a simple standalone application, not a formal napari plugin. The flat file structure contains:
- Two main Python scripts
- Conda environment specification
- Setup automation script

### Common Issues
- **MRC file corruption**: Application handles this with permissive mode
- **Memory usage**: Large datasets require proper chunking via Dask
- **Display issues**: GUI may fail in headless environments
- **Float precision**: Handles float16/float32 conversions for processing

### Scientific Domain Context
This tool is specialized for cryo-EM structural biology workflows, specifically:
- Amyloid fibril annotation in micrographs  
- Manual ground truth generation for ML training
- Interactive quality control of cryo-EM data
- Resolution-based filtering using scientific units (Angstroms)