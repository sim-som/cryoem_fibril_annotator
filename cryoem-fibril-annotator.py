#!/usr/bin/env python3
"""
Napari-based tool for annotating fibrils in cryo-EM micrographs.
Features:
- Lazy loading of MRC stacks using Dask
- Interactive Butterworth lowpass filter with angstrom-based threshold
- Manual fibril annotation via shapes layer
"""

import numpy as np
import napari
import dask.array as da
from dask import delayed
import mrcfile
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
from magicgui import magicgui
from napari.types import ImageData
from skimage.filters import butterworth
from glob import glob
import sys
from collections import Counter
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning)


# Integrated functions from cryoem-file-finder.py
def find_corresponding_file(
    input_file: str, 
    target_folder: str, 
    micrograph_suffixes: List[str] = None,
    power_spectrum_suffixes: List[str] = None
) -> Optional[str]:
    """
    Find the corresponding cryo-EM data file between micrographs and power spectra.
    """
    if micrograph_suffixes is None:
        micrograph_suffixes = [
            '_doseweighted.mrc',
            '_aligned.mrc', 
            '_motion_corrected.mrc',
            '_patch_aligned_doseweighted.mrc',
            '_fractions_patch_aligned_doseweighted.mrc'
        ]
    
    if power_spectrum_suffixes is None:
        power_spectrum_suffixes = [
            '_ctf_diag_2D.mrc',
            '_ctf_2D.mrc',
            '_power_spectrum.mrc',
            '_ps.mrc',
            '_ctf_diag.mrc'
        ]
    
    input_path = Path(input_file)
    input_name = input_path.name
    target_path = Path(target_folder)
    
    if not target_path.exists():
        return None
    
    # Determine input file type and extract base name
    base_name = None
    target_suffixes = []
    
    # Check if input is a micrograph
    for suffix in micrograph_suffixes:
        if input_name.endswith(suffix):
            base_name = input_name[:-len(suffix)]
            target_suffixes = power_spectrum_suffixes
            break
    
    # Check if input is a power spectrum
    if base_name is None:
        for suffix in power_spectrum_suffixes:
            if input_name.endswith(suffix):
                base_name = input_name[:-len(suffix)]
                target_suffixes = micrograph_suffixes
                break
    
    if base_name is None:
        return None
    
    # Search for corresponding file with any of the target suffixes
    for suffix in target_suffixes:
        candidate_file = target_path / (base_name + suffix)
        if candidate_file.exists():
            return str(candidate_file)
    
    return None


class CryoEMFibrilAnnotator:
    """Main class for cryo-EM fibril annotation with interactive filtering."""
    
    def __init__(self, mrc_files: List[str], pixel_size: float = None, ps_files: List[str] = None):
        """
        Initialize the annotator.
        
        Parameters
        ----------
        mrc_files : List[str]
            List of paths to MRC files
        pixel_size : float, optional
            Pixel size in Angstroms. If None, will try to read from MRC header
        ps_files : List[str], optional
            List of paths to power spectrum MRC files
        """
        self.mrc_files = sorted(mrc_files)
        self.ps_files = sorted(ps_files) if ps_files else None
        self.pixel_size = pixel_size
        self.viewer = None
        self.image_layer = None
        self.filtered_layer = None
        self.ps_layer = None  # Power spectrum layer
        self.shapes_layers = {}  # Dictionary to store multiple annotation layers
        self.active_shapes_layer = None  # Currently active annotation layer
        self.original_stack = None
        self.ps_stack = None  # Power spectrum stack
        self.current_filter_threshold = None
        self.filter_order = 4  # Default Butterworth filter order
        self.force_permissive = False  # Whether to force permissive mode
        
    def load_mrc_lazy(self, filepath: str, shape:Tuple, dtype) -> Tuple[Optional[da.Array], float]:
        """
        Lazily load a single MRC file.
        
        Parameters
        ----------
        filepath : str
            Path to MRC file
            
        Returns
        -------
        data : dask.array or None
            Lazy array of the image data, or None if file is corrupted
        pixel_size : float
            Pixel size in Angstroms
        """
        # Determine whether to use permissive mode
        use_permissive = self.force_permissive
                
        @delayed
        def _load_mrc(path, permissive):
            try:
                with mrcfile.open(path, mode='r', permissive=permissive) as mrc:
                    if mrc.data is None:
                        raise ValueError(f"No data in {path}")
                    data = mrc.data.copy()
            except Exception as e:
                if not permissive:
                    # Fallback to permissive mode for actual loading
                    with mrcfile.open(path, mode='r', permissive=True) as mrc:
                        if mrc.data is None:
                            raise ValueError(f"No data in {path}")
                        data = mrc.data.copy()
                else:
                    raise e
            
            # Handle different dimensionalities
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            return data
        
        # Handle different dimensionalities
        if len(shape) == 2:
            shape = (1,) + shape
            
        # Create lazy array
        delayed_array = _load_mrc(filepath, use_permissive)
        lazy_array = da.from_delayed(delayed_array, shape=shape, dtype=dtype)
        
        return lazy_array 
    
    def create_stack(self) -> Tuple[da.Array, float]:
        """
        Create a lazy stack from all MRC files.
        
        Returns
        -------
        stack : dask.array
            Lazy stack of all micrographs
        pixel_size : float
            Pixel size in Angstroms
        """
        arrays = []
        valid_files = []

        # Get shape and dtype from first micrograph (assume this is constant for all micrographs)
        with mrcfile.open(self.mrc_files[0]) as mrc:
            mic_shape = mrc.data.shape
            mic_dtype = mrc.data.dtype

            # Also try to get pixel size from header:
            if self.pixel_size is None:
                voxel_size = mrc.voxel_size
                if voxel_size.x > 0:
                    pixel_size = float(voxel_size.x)
                else:
                    print(f"Warning: Could not read pixel size from {filepath}, using 1.0 Å")
                    pixel_size = 1.0  # Default to 1 Angstrom 
        
        # Iterate over all file paths and lazy read images as dask arrays:
        for filepath in tqdm(self.mrc_files, desc=f"Loading {len(self.mrc_files)} MRC files..."):
            arr = self.load_mrc_lazy(filepath, shape=mic_shape, dtype=mic_dtype)
            if arr is not None:
                arrays.append(arr)
                valid_files.append(filepath)
            else:
                print(f"Skipped: {filepath}")
        
        if not arrays:
            raise ValueError("No valid MRC files could be loaded!")
        
        print(f"Successfully loaded {len(arrays)} out of {len(self.mrc_files)} files")
        
        # Update the file list to only include valid files
        self.mrc_files = valid_files
        
        # Concatenate along first axis
        stack = da.concatenate(arrays, axis=0)
        
        # If individual files were 2D, squeeze the extra dimension
        if stack.ndim == 4 and stack.shape[1] == 1:
            stack = stack.squeeze(axis=1)
        
        print(f"Stack shape: {stack.shape}")
        print(f"Stack dtype: {stack.dtype}")
        print(f"Estimated size: {stack.nbytes / 1e9:.2f} GB")
        
        return stack, pixel_size
    
    def create_ps_stack(self) -> Tuple[Optional[da.Array], float]:
        """
        Create a lazy stack from all power spectrum MRC files.
        
        Returns
        -------
        stack : dask.array or None
            Lazy stack of all power spectra, or None if no PS files
        pixel_size : float
            Pixel size in Angstroms (from micrographs)
        """
        if not self.ps_files:
            return None, self.pixel_size
            
        arrays = []
        valid_files = []

        # Get shape and dtype from first powerspectrum (assume this is constant for all power spectrums)
        with mrcfile.open(self.ps_files[0]) as mrc:
            ps_shape = mrc.data.shape
            ps_dtype = mrc.data.dtype
            # (pixel size for powerspectra not relevant)

        for filepath in tqdm(self.ps_files, desc=f"Loading {len(self.ps_files)} power spectrum files..."):
            arr = self.load_mrc_lazy(filepath, shape=ps_shape, dtype=ps_dtype) 
            if arr is not None:
                arrays.append(arr)
                valid_files.append(filepath)
            else:
                print(f"Skipped PS: {filepath}")
        
        if not arrays:
            print("Warning: No valid power spectrum files could be loaded!")
            return None, self.pixel_size
        
        print(f"Successfully loaded {len(arrays)} out of {len(self.ps_files)} PS files")
        
        # Update the PS file list to only include valid files
        self.ps_files = valid_files
        
        # Concatenate along first axis
        stack = da.concatenate(arrays, axis=0)
        
        # If individual files were 2D, squeeze the extra dimension
        if stack.ndim == 4 and stack.shape[1] == 1:
            stack = stack.squeeze(axis=1)
        
        print(f"PS Stack shape: {stack.shape}")
        print(f"PS Stack dtype: {stack.dtype}")
        print(f"PS Estimated size: {stack.nbytes / 1e9:.2f} GB")
        
        return stack
    
    def create_filtered_stack_lazy(self, threshold_angstrom: float, order: int = 4) -> da.Array:
        """
        Create a lazy filtered version of the stack using Dask.
        
        Parameters
        ----------
        threshold_angstrom : float
            Filter threshold in Angstroms
        order : int
            Butterworth filter order
            
        Returns
        -------
        filtered_stack : dask.array
            Lazy filtered stack
        """
        def filter_wrapper(block, block_id=None):
            """Wrapper to apply filter to each 2D slice in a block."""
            _ = block_id  # Unused parameter
            # Store original dtype
            original_dtype = block.dtype
            
            if block.ndim == 3:
                # Process each slice in the block
                # Use float32 for processing to avoid overflow
                filtered_block = np.empty(block.shape, dtype=np.float32)
                for i in range(block.shape[0]):
                    filtered_block[i] = self.apply_lowpass_filter(
                        block[i], threshold_angstrom, order
                    ).astype(np.float32)
                
                # Convert back to original dtype if needed
                if original_dtype == np.float16:
                    filtered_block = np.clip(filtered_block, 
                                            np.finfo(np.float16).min, 
                                            np.finfo(np.float16).max)
                    return filtered_block.astype(original_dtype)
                else:
                    return filtered_block.astype(original_dtype)
            else:
                # Single 2D image
                return self.apply_lowpass_filter(block, threshold_angstrom, order)
        
        # Apply filter lazily using map_blocks
        filtered = da.map_blocks(
            filter_wrapper,
            self.original_stack,
            dtype=self.original_stack.dtype,
            chunks=self.original_stack.chunks
        )
        
        return filtered
    
    def create_annotation_layer(self, name: str, color: str = 'red') -> None:
        """
        Create a new annotation layer for a specific fibril type.
        
        Parameters
        ----------
        name : str
            Name of the annotation layer
        color : str
            Color for the annotations (default: 'red')
        """
        if name in self.shapes_layers:
            print(f"Warning: Layer '{name}' already exists")
            return
            
        # Calculate fibril width
        fibril_width = 200
        if self.pixel_size:
            fibril_width = fibril_width / self.pixel_size

        # Set ndim to match the image stack (3D if stack, 2D if single image)
        if len(self.original_stack.shape) > 2:
            # 3D stack - annotations are per frame
            shapes_layer = self.viewer.add_shapes(
                name=f'{name} Annotations',
                ndim=3,  # This ensures annotations are per-frame
                shape_type='line',  # Lines for fibril annotation
                edge_color=color,
                edge_width=fibril_width,
                face_color='transparent'
            )
        else:
            # Single 2D image
            shapes_layer = self.viewer.add_shapes(
                name=f'{name} Annotations',
                ndim=2,
                shape_type='line',
                edge_color=color,
                edge_width=fibril_width,
                face_color='transparent'
            )
        
        self.shapes_layers[name] = shapes_layer
        self.active_shapes_layer = shapes_layer
        print(f"Created annotation layer: {name}")
    
    def apply_lowpass_filter(self, image: np.ndarray, threshold_angstrom: float, order: int = 4) -> np.ndarray:
        """
        Apply Butterworth lowpass filter to image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        threshold_angstrom : float
            Filter threshold in Angstroms (resolution cutoff)
        order : int
            Order of the Butterworth filter (higher = sharper cutoff)
            
        Returns
        -------
        filtered : np.ndarray
            Filtered image
        """
        if threshold_angstrom <= 0:
            return image
        
        # Store original dtype
        original_dtype = image.dtype
        
        # Convert to float32 for processing (butterworth and float16 don't mix well)
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Convert resolution in Angstroms to frequency cutoff
        # Calculate the Nyquist frequency (highest frequency = 1/(2*pixel_size))
        nyquist_resolution = 2 * self.pixel_size  # in Angstroms
        
        # Calculate cutoff frequency as fraction of Nyquist
        # Lower resolution (higher Angstrom value) = lower frequency cutoff
        cutoff_freq_normalized = nyquist_resolution / threshold_angstrom
        
        # Ensure cutoff is within valid range (0, 1]
        cutoff_freq_normalized = np.clip(cutoff_freq_normalized, 0.01, 1.0)
        
        try:
            # Apply Butterworth lowpass filter
            filtered = butterworth(
                image,
                cutoff_frequency_ratio=cutoff_freq_normalized,
                order=order,
                high_pass=False  # We want lowpass
            )
        except Exception as e:
            print(f"Warning: Butterworth filter failed ({e}), returning original image")
            filtered = image
        
        # Convert back to original dtype if it was float16
        if original_dtype == np.float16:
            # Clip to float16 range to avoid overflow
            filtered = np.clip(filtered, np.finfo(np.float16).min, np.finfo(np.float16).max)
            filtered = filtered.astype(original_dtype)
        
        return filtered
    
    def update_filter(self, threshold_angstrom: float):
        """
        Update the lowpass filter threshold and refresh display.
        
        Parameters
        ----------
        threshold_angstrom : float
            New filter threshold in Angstroms
        """
        if self.viewer is None or self.image_layer is None:
            return
        
        self.current_filter_threshold = threshold_angstrom
        
        if threshold_angstrom > 0:
            # Create or update filtered layer using lazy filtering
            try:
                # Create lazy filtered stack
                filtered_stack = self.create_filtered_stack_lazy(threshold_angstrom, self.filter_order)
                
                if self.filtered_layer is not None:
                    # Update existing layer with new lazy stack
                    self.filtered_layer.data = filtered_stack
                else:
                    # Create new filtered layer
                    self.filtered_layer = self.viewer.add_image(
                        filtered_stack,
                        name='Filtered',
                        colormap='gray',
                        contrast_limits=self.image_layer.contrast_limits,
                        blending='additive',
                        opacity=1.0
                    )
                
                # Hide original, show filtered
                self.image_layer.visible = False
                self.filtered_layer.visible = True
                
            except Exception as e:
                print(f"Error applying filter: {e}")
                # Fallback to showing original
                if self.filtered_layer is not None:
                    self.filtered_layer.visible = False
                self.image_layer.visible = True
        else:
            # No filtering - show original
            if self.filtered_layer is not None:
                self.filtered_layer.visible = False
            self.image_layer.visible = True
    
    def run(self):
        """Launch the napari viewer with annotation interface."""
        # Create lazy stack
        self.original_stack, self.pixel_size = self.create_stack()
        
        # Create power spectrum stack if available
        if self.ps_files:
            print(f"Loading power spectra for {len(self.ps_files)} files...")
            self.ps_stack = self.create_ps_stack()
        else:
            print("No power spectrum files provided")
        
        # Initialize viewer
        self.viewer = napari.Viewer(title='Cryo-EM Fibril Annotator')
        
        # Calculate contrast limits from first image
        print("Calculating contrast limits...")
        first_image = self.original_stack[0].compute()
        
        # Convert to float32 for percentile calculation to avoid overflow with float16
        if first_image.dtype == np.float16:
            first_image_f32 = first_image.astype(np.float32)
        else:
            first_image_f32 = first_image
        
        # Calculate percentiles
        contrast_min = float(np.percentile(first_image_f32, 1))
        contrast_max = float(np.percentile(first_image_f32, 99))
        
        # Handle edge cases
        if np.isnan(contrast_min) or np.isnan(contrast_max) or contrast_min == contrast_max:
            print("Warning: Could not calculate contrast limits from percentiles, using min/max")
            contrast_min = float(np.min(first_image_f32))
            contrast_max = float(np.max(first_image_f32))
            
            # If still having issues, use a default range
            if np.isnan(contrast_min) or np.isnan(contrast_max) or contrast_min == contrast_max:
                print("Warning: Using default contrast range")
                contrast_min = -1.0
                contrast_max = 1.0
        
        print(f"Contrast limits: [{contrast_min:.2f}, {contrast_max:.2f}]")
        
        # Add image layer
        self.image_layer = self.viewer.add_image(
            self.original_stack,
            name='Micrographs',
            colormap='gray',
            contrast_limits=(contrast_min, contrast_max)
        )
        
        # Add power spectrum layer if available
        if self.ps_stack is not None:
            # Calculate contrast limits for power spectrum
            print("Calculating power spectrum contrast limits...")
            first_ps = self.ps_stack[0].compute()
            
            # Convert to float32 for percentile calculation
            if first_ps.dtype == np.float16:
                first_ps_f32 = first_ps.astype(np.float32)
            else:
                first_ps_f32 = first_ps
            
            # Power spectra often have high dynamic range, use more conservative percentiles
            ps_contrast_min = float(np.percentile(first_ps_f32, 5))
            ps_contrast_max = float(np.percentile(first_ps_f32, 95))
            
            # Handle edge cases
            if np.isnan(ps_contrast_min) or np.isnan(ps_contrast_max) or ps_contrast_min == ps_contrast_max:
                ps_contrast_min = float(np.min(first_ps_f32))
                ps_contrast_max = float(np.max(first_ps_f32))
            
            print(f"Power spectrum contrast limits: [{ps_contrast_min:.2f}, {ps_contrast_max:.2f}]")
            
            self.ps_layer = self.viewer.add_image(
                self.ps_stack,
                name='Power Spectra',
                colormap='viridis',  # Different colormap for distinction
                contrast_limits=(ps_contrast_min, ps_contrast_max),
                visible=False  # Start hidden
            )
            
            print(f"Added power spectrum layer with {len(self.ps_files)} files")
        
        # Create default annotation layer
        self.create_annotation_layer('Default', 'red')
        
        # Create display control widget
        @magicgui(
            auto_call=True,
            show_power_spectra={'widget_type': 'CheckBox',
                               'value': False,
                               'label': 'Show Power Spectra',
                               'tooltip': 'Toggle power spectrum display'}
        )
        def display_controls(show_power_spectra: bool = False):
            """Control panel for display options."""
            if self.ps_layer is not None:
                self.ps_layer.visible = show_power_spectra
                if show_power_spectra:
                    self.viewer.status = 'Power spectra visible'
                else:
                    self.viewer.status = 'Power spectra hidden'
        
        # Create filter control widget
        @magicgui(
            auto_call=True,
            lowpass_threshold={'widget_type': 'FloatSlider', 
                             'min': 0, 
                             'max': 100,
                             'value': 0,
                             'label': 'Lowpass Filter (Å)',
                             'tooltip': 'Resolution cutoff in Angstroms (0 = no filter)'},
            filter_order={'widget_type': 'IntSlider',
                         'min': 1,
                         'max': 10,
                         'value': 4,
                         'label': 'Filter Order',
                         'tooltip': 'Butterworth filter order (higher = sharper cutoff)'}
        )
        def filter_controls(lowpass_threshold: float = 0, filter_order: int = 4):
            """Control panel for Butterworth lowpass filter."""
            self.filter_order = filter_order
            self.update_filter(lowpass_threshold)
            if lowpass_threshold > 0:
                self.viewer.status = f'Butterworth lowpass: {lowpass_threshold:.1f} Å (order={filter_order})'
            else:
                self.viewer.status = 'No filter applied'
        
        # Create load annotations widget
        @magicgui(
            call_button='Load Annotations',
            filename={'widget_type': 'FileEdit', 
                     'mode': 'r',
                     'value': 'fibril_annotations.npy',
                     'label': 'Load from:'}
        )
        def load_annotations(filename: Path = Path('fibril_annotations.npy')):
            """Load annotations from file into a new layer."""
            try:
                if not filename.exists():
                    self.viewer.status = f'File not found: {filename}'
                    return
                
                # Load annotations data
                annotations = np.load(filename, allow_pickle=True).item()
                
                if not isinstance(annotations, dict):
                    self.viewer.status = 'Invalid annotation file format'
                    return
                
                # Validate required fields
                required_fields = ['shapes', 'pixel_size', 'mrc_files', 'ndim']
                missing_fields = [field for field in required_fields if field not in annotations]
                if missing_fields:
                    self.viewer.status = f'Missing fields in annotation file: {missing_fields}'
                    return
                
                # Check if pixel size matches (with tolerance)
                loaded_pixel_size = annotations['pixel_size']
                if self.pixel_size and abs(loaded_pixel_size - self.pixel_size) > 0.01:
                    print(f"Warning: Pixel size mismatch! Current: {self.pixel_size} Å, Loaded: {loaded_pixel_size} Å")
                    self.viewer.status = f'Warning: Pixel size mismatch (loaded: {loaded_pixel_size} Å)'
                
                # Check dimensionality compatibility
                expected_ndim = 3 if len(self.original_stack.shape) > 2 else 2
                loaded_ndim = annotations['ndim']
                if loaded_ndim != expected_ndim:
                    self.viewer.status = f'Dimension mismatch: expected {expected_ndim}D, loaded {loaded_ndim}D'
                    return
                    
                # Load the shapes data
                shapes_data = annotations['shapes']
                
                if len(shapes_data) > 0:
                    # Determine layer name and color
                    layer_name = annotations.get('layer_name', f'Loaded_{filename.stem}')
                    edge_color = annotations.get('edge_color', 'blue')  # Default to blue for loaded layers
                    
                    # Remove ' Annotations' suffix if present to avoid duplication
                    if layer_name.endswith(' Annotations'):
                        layer_name = layer_name[:-12]
                    
                    # Create unique layer name if it already exists
                    original_layer_name = layer_name
                    counter = 1
                    while layer_name in self.shapes_layers:
                        layer_name = f"{original_layer_name}_{counter}"
                        counter += 1
                    
                    # Create new annotation layer
                    self.create_annotation_layer(layer_name, edge_color)
                    new_layer = self.shapes_layers[layer_name]
                    
                    # Set the annotation data
                    new_layer.data = shapes_data
                    
                    # Load properties if available
                    if 'properties' in annotations and annotations['properties']:
                        new_layer.properties = annotations['properties']
                    
                    # Ensure proper edge width for all loaded annotations
                    # Calculate the proper fibril width
                    fibril_width = 200
                    if self.pixel_size:
                        fibril_width = fibril_width / self.pixel_size
                    
                    # Set edge width for the layer (affects new annotations)
                    new_layer.edge_width = fibril_width
                    
                    # Also update edge widths for all existing annotations
                    if len(shapes_data) > 0:
                        # Create edge_width array for all shapes
                        edge_widths = [fibril_width] * len(shapes_data)
                        new_layer.edge_width = edge_widths
                    
                    # Print summary for 3D annotations
                    if loaded_ndim == 3 and 'frame_indices' in annotations:
                        frame_indices = annotations['frame_indices']
                        frame_counts = Counter(frame_indices)
                        total_frames_with_annotations = len(frame_counts)
                        
                        print(f"Loaded annotations summary for {layer_name}:")
                        print(f"  Total annotations: {len(shapes_data)}")
                        print(f"  Frames with annotations: {total_frames_with_annotations}")
                        for frame, count in sorted(frame_counts.items()):
                            if frame < len(self.mrc_files):
                                print(f"    Frame {frame} ({Path(self.mrc_files[frame]).name}): {count} annotations")
                    
                    self.viewer.status = f'Loaded {len(shapes_data)} annotations into new layer: {layer_name}'
                    print(f"Successfully loaded {len(shapes_data)} annotations into layer: {layer_name}")
                else:
                    self.viewer.status = f'No annotations found in {filename}'
                    
            except Exception as e:
                error_msg = f'Error loading annotations: {str(e)}'
                self.viewer.status = error_msg
                print(error_msg)
        
        # Create layer management widget
        @magicgui(
            call_button='Create New Layer',
            layer_name={'widget_type': 'LineEdit',
                       'value': 'Aβ42',
                       'label': 'Layer Name:'},
            layer_color={'widget_type': 'ComboBox',
                        'choices': ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple'],
                        'value': 'green',
                        'label': 'Color:'}
        )
        def layer_management(layer_name: str = 'Aβ42', layer_color: str = 'green'):
            """Create a new annotation layer for a specific fibril type."""
            if not layer_name.strip():
                self.viewer.status = 'Please enter a layer name'
                return
                
            layer_name = layer_name.strip()
            
            if layer_name in self.shapes_layers:
                self.viewer.status = f'Layer "{layer_name}" already exists'
                return
                
            try:
                self.create_annotation_layer(layer_name, layer_color)
                self.viewer.status = f'Created new annotation layer: {layer_name} ({layer_color})'
                print(f"Created new annotation layer: {layer_name} with color {layer_color}")
            except Exception as e:
                error_msg = f'Error creating layer: {str(e)}'
                self.viewer.status = error_msg
                print(error_msg)
        
        # Create annotation controls widget
        @magicgui(
            call_button='Save Annotations',
            filename={'widget_type': 'FileEdit', 
                     'mode': 'w',
                     'value': 'fibril_annotations.npy',
                     'label': 'Save to:'}
        )
        def annotation_controls(filename: Path = Path('fibril_annotations.npy')):
            """Save annotations from the currently selected layer to file."""
            # Find the currently selected shapes layer
            selected_layer = None
            layer_name = None
            
            # Check if there are any selected layers in the layer list
            if len(self.viewer.layers.selection) > 0:
                # Get the first selected layer
                for layer in self.viewer.layers.selection:
                    # Check if it's a shapes layer (has shape_type attribute)
                    if hasattr(layer, 'shape_type') and hasattr(layer, 'data'):
                        selected_layer = layer
                        layer_name = layer.name
                        break
            
            # If no shapes layer is selected, use the active shapes layer as fallback
            if selected_layer is None and self.active_shapes_layer is not None:
                selected_layer = self.active_shapes_layer
                layer_name = selected_layer.name
                print("No layer selected, using active layer as fallback")
            
            if selected_layer is not None and len(selected_layer.data) > 0:
                print(f"Saving layer: {layer_name} with {len(selected_layer.data)} annotations")
                # Get annotation data
                annotations = {
                    'shapes': selected_layer.data,
                    'shape_types': selected_layer.shape_type,
                    'properties': selected_layer.properties,  # Any additional properties
                    'pixel_size': self.pixel_size,
                    'mrc_files': self.mrc_files,
                    'ndim': selected_layer.ndim,  # Important for reloading
                    'layer_name': layer_name,  # Store the layer name
                    'edge_color': selected_layer.edge_color,  # Store the color
                }
                
                # For 3D annotations, include frame information
                if selected_layer.ndim == 3:
                    # Extract which frame each annotation belongs to
                    frame_indices = []
                    for shape in selected_layer.data:
                        # First coordinate is the frame index in 3D
                        if len(shape) > 0:
                            frame_idx = int(shape[0, 0])  # z-coordinate
                            frame_indices.append(frame_idx)
                    annotations['frame_indices'] = frame_indices
                    
                    # Count annotations per frame
                    from collections import Counter
                    frame_counts = Counter(frame_indices)
                    total_frames_with_annotations = len(frame_counts)
                    
                    print(f"Annotations summary for {layer_name}:")
                    print(f"  Total annotations: {len(selected_layer.data)}")
                    print(f"  Frames with annotations: {total_frames_with_annotations}/{len(self.mrc_files)}")
                    for frame, count in sorted(frame_counts.items()):
                        if frame < len(self.mrc_files):
                            print(f"    Frame {frame} ({Path(self.mrc_files[frame]).name}): {count} annotations")
                
                # Save to file
                np.save(filename, annotations, allow_pickle=True)
                self.viewer.status = f'Saved {len(selected_layer.data)} annotations from {layer_name} to {filename}'
                print(f"\nAnnotations from {layer_name} saved to {filename}")
            else:
                if selected_layer is None:
                    self.viewer.status = 'No annotation layer selected. Please select a shapes layer in the layer list.'
                    print("No annotation layer selected for saving")
                else:
                    self.viewer.status = 'Selected layer has no annotations to save'
                    print(f"Layer {layer_name} has no annotations to save")
        
        # Add widgets to viewer
        if self.ps_stack is not None:
            self.viewer.window.add_dock_widget(display_controls, area='right', name='Display Controls')
        self.viewer.window.add_dock_widget(filter_controls, area='right', name='Filter Controls')
        self.viewer.window.add_dock_widget(layer_management, area='right', name='Layer Management')
        self.viewer.window.add_dock_widget(load_annotations, area='right', name='Load Annotations')
        self.viewer.window.add_dock_widget(annotation_controls, area='right', name='Save Annotations')
        
        # Add usage instructions
        instructions = """
        FIBRIL ANNOTATION INSTRUCTIONS:
        
        1. Display Controls:
           - Toggle 'Show Power Spectra' to view corresponding 2D power spectra
           - Power spectra are automatically synchronized with micrograph navigation
        
        2. Use slider to adjust Butterworth lowpass filter (0 = no filter)
           - Resolution cutoff in Angstroms
           - Filter order controls sharpness (higher = sharper cutoff)
        
        3. Layer Management:
           - Use 'Layer Management' to create new annotation layers for different fibril types
           - Default suggestions: Aβ42, Tau, α-synuclein, etc.
           - Each layer can have a different color for easy identification
           - Select the appropriate layer in the layer list before annotating
        
        4. Annotation tools:
           - Press 'L' for line tool (straight fibril segments)
           - Press 'Shift+L' for polyline tool (multi-segment fibrils)
           - Click to set start point, click again for end point
           - For polylines: keep clicking to add segments, double-click or Enter to finish
        
        5. Editing tools:
           - Press 'A' for selection tool to edit/delete annotations
           - Press 'D' to delete selected annotations
           - Press Escape to cancel current annotation
           - Press 'M' for pan/zoom mode
        
        6. Save/Load:
           - Use 'Save Annotations' to export the currently selected layer
           - Each layer should be saved to a separate .npy file (e.g., Ab42_annotations.npy, Tau_annotations.npy)
           - Use 'Load Annotations' to load annotations into a new layer
           - Loading preserves the original layer name and color from the file
        
        Navigation:
        - Scroll: zoom in/out
        - Click+drag: pan (when in pan mode 'M')
        - Slider at bottom: navigate between micrographs
        - Arrow keys: fine navigation between frames
        - Power spectra automatically follow micrograph navigation
        
        Note: Each micrograph has its own independent annotations for each layer
        Multiple annotation layers allow you to classify different fibril types
        """
        
        print(instructions)
        self.viewer.status = "Ready for annotation. Press 'L' for lines or 'Shift+L' for polylines to trace fibrils."
        
        # Start viewer
        napari.run()


def main():
    """Main entry point for the script.
    
    Required dependencies:
        pip install napari[all] dask mrcfile scikit-image magicgui
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Annotate fibrils in cryo-EM micrographs')
    # micrograph mrc files:
    parser.add_argument('mic_dir', help='Location of MRC micrographs')
    parser.add_argument('--glob_pattern', default="*_fractions_patch_aligned_doseweighted.mrc", help='Search pattern (default: "*_fractions_patch_aligned_doseweighted.mrc")')
    # TODO add option to select commond file endings (*.mrc, *_fracitions.mrc, etc.)

    # Optional: Add location of 2D powerspectra, that were already calculated during CTF estimation / preproscessing:
    parser.add_argument("--ps_dir", default=None, help="Directory path with precalculated 2D powerspectra (optinal but helpful for amyloid fibril picking)")
    parser.add_argument("--ps_glob", default="*_ctf_diag_2D.mrc", help='Search pattern for 2D powerspectra files (default: "*_ctf_diag_2D.mrc")')

    parser.add_argument('--pixel-size', type=float, default=None,
                       help='Pixel size in Angstroms (if not in MRC header)')
    parser.add_argument('--permissive', action='store_true',
                       help='Force permissive mode for corrupted MRC files (use with caution)')
    
    args = parser.parse_args()
    mic_dir = Path(args.mic_dir)
    assert mic_dir.exists()
    assert mic_dir.is_dir()
    
    # Get list of micrograph mrc files:
    print(f"Searching vor mics in {mic_dir}/{args.glob_pattern} ...")
    mic_files = [f for f in mic_dir.glob(args.glob_pattern)]
    
    if not mic_files:
        print("Error: No MRC files found!")
        sys.exit(1)

    num_mics = len(mic_files)
    print(f"Found {num_mics} micrographs")
    
    # Check files exist
    mic_files = [f for f in tqdm(mic_files, desc="Checking if files exist") if Path(f).exists()]
    if not mic_files:
        print("Error: No valid MRC files found!")
        sys.exit(1)
    
    print(f"Found {len(mic_files)} MRC files")

    # if given get list of 2D powerspectra files:
    ps_files = None
    if args.ps_dir:
        ps_dir = Path(args.ps_dir)
        if not ps_dir.exists() or not ps_dir.is_dir():
            print(f"Warning: Power spectrum directory {ps_dir} does not exist or is not a directory")
            print("Continuing without power spectra...")
        else:
            # Try to find corresponding power spectrum files
            ps_files = []
            missing_ps = []
            
            for mic_file in tqdm(mic_files, desc="Looking for corresponding power spectra files"):
                corresponding_ps = find_corresponding_file(str(mic_file), str(ps_dir))
                if corresponding_ps and Path(corresponding_ps).exists():
                    ps_files.append(corresponding_ps)
                else:
                    missing_ps.append(mic_file)
                    # Try direct glob pattern match as fallback
                    ps_candidates = [f for f in ps_dir.glob(args.ps_glob)]
                    # Find PS file with matching base name
                    mic_base = Path(mic_file).stem
                    for candidate in ps_candidates:
                        if mic_base.split('_fractions')[0] in candidate.stem:
                            ps_files.append(str(candidate))
                            missing_ps.pop()
                            break
            
            if len(ps_files) != len(mic_files):
                print(f"Warning: Found {len(ps_files)} power spectra for {len(mic_files)} micrographs")
                if missing_ps:
                    print(f"Missing power spectra for {len(missing_ps)} micrographs:")
                    for missing in missing_ps[:5]:  # Show first 5
                        print(f"  {Path(missing).name}")
                    if len(missing_ps) > 5:
                        print(f"  ... and {len(missing_ps) - 5} more")
                
                # Only use matching pairs
                if len(ps_files) > 0:
                    print(f"Using {len(ps_files)} matched micrograph-power spectrum pairs")
                    # Truncate mic_files to match ps_files length
                    mic_files = mic_files[:len(ps_files)]
                else:
                    print("No matching power spectra found, continuing without power spectra")
                    ps_files = None
            else:
                print(f"Found matching power spectra for all {len(ps_files)} micrographs")

    
    # Create and run annotator
    annotator = CryoEMFibrilAnnotator(mic_files, pixel_size=args.pixel_size, ps_files=ps_files)
    
    # Set permissive mode if requested
    if args.permissive:
        print("Warning: Running in permissive mode - corrupted files may cause issues")
        annotator.force_permissive = True
    else:
        annotator.force_permissive = False
    
    annotator.run()


if __name__ == '__main__':
    main()