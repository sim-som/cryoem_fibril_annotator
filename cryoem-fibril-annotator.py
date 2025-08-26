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

warnings.filterwarnings('ignore', category=FutureWarning)


class CryoEMFibrilAnnotator:
    """Main class for cryo-EM fibril annotation with interactive filtering."""
    
    def __init__(self, mrc_files: List[str], pixel_size: float = None):
        """
        Initialize the annotator.
        
        Parameters
        ----------
        mrc_files : List[str]
            List of paths to MRC files
        pixel_size : float, optional
            Pixel size in Angstroms. If None, will try to read from MRC header
        """
        self.mrc_files = sorted(mrc_files)
        self.pixel_size = pixel_size
        self.viewer = None
        self.image_layer = None
        self.filtered_layer = None
        self.shapes_layer = None
        self.original_stack = None
        self.current_filter_threshold = None
        self.filter_order = 4  # Default Butterworth filter order
        self.force_permissive = False  # Whether to force permissive mode
        
    def load_mrc_lazy(self, filepath: str) -> Tuple[Optional[da.Array], float]:
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
        
        try:
            # First try to open with chosen mode
            with mrcfile.open(filepath, mode='r', permissive=use_permissive) as mrc:
                if mrc.data is None:
                    print(f"Warning: No data in {filepath}, skipping")
                    return None, 1.0
                    
                shape = mrc.data.shape
                dtype = mrc.data.dtype
                
                # Try to get pixel size from header
                if self.pixel_size is None:
                    voxel_size = mrc.voxel_size
                    if voxel_size.x > 0:
                        pixel_size = float(voxel_size.x)
                    else:
                        print(f"Warning: Could not read pixel size from {filepath}, using 1.0 Å")
                        pixel_size = 1.0  # Default to 1 Angstrom
                else:
                    pixel_size = self.pixel_size
                    
        except Exception as e:
            if not use_permissive:
                # Try with permissive mode as fallback
                try:
                    print(f"Warning: Standard MRC read failed for {filepath}: {e}")
                    print(f"Trying permissive mode...")
                    with mrcfile.open(filepath, mode='r', permissive=True) as mrc:
                        if mrc.data is None:
                            print(f"Error: Could not read data from {filepath}, skipping")
                            return None, 1.0
                            
                        shape = mrc.data.shape
                        dtype = mrc.data.dtype
                        pixel_size = self.pixel_size if self.pixel_size else 1.0
                        use_permissive = True  # Remember to use permissive for loading
                        
                except Exception as e2:
                    print(f"Error: Could not read {filepath}: {e2}, skipping")
                    return None, 1.0
            else:
                print(f"Error: Could not read {filepath} even in permissive mode: {e}, skipping")
                return None, 1.0
        
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
        
        return lazy_array, pixel_size
    
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
        pixel_sizes = []
        valid_files = []
        
        print(f"Loading {len(self.mrc_files)} MRC files...")
        
        for filepath in self.mrc_files:
            arr, pix_size = self.load_mrc_lazy(filepath)
            if arr is not None:
                arrays.append(arr)
                pixel_sizes.append(pix_size)
                valid_files.append(filepath)
            else:
                print(f"Skipped: {filepath}")
        
        if not arrays:
            raise ValueError("No valid MRC files could be loaded!")
        
        print(f"Successfully loaded {len(arrays)} out of {len(self.mrc_files)} files")
        
        # Update the file list to only include valid files
        self.mrc_files = valid_files
        
        # Check if all pixel sizes are the same
        if len(set(pixel_sizes)) > 1:
            print("Warning: Different pixel sizes detected across files!")
            print(f"Pixel sizes: {set(pixel_sizes)}")
            pixel_size = pixel_sizes[0]
            print(f"Using pixel size from first file: {pixel_size} Å")
        else:
            pixel_size = pixel_sizes[0]
            print(f"Pixel size: {pixel_size} Å")
        
        # Concatenate along first axis
        stack = da.concatenate(arrays, axis=0)
        
        # If individual files were 2D, squeeze the extra dimension
        if stack.ndim == 4 and stack.shape[1] == 1:
            stack = stack.squeeze(axis=1)
        
        print(f"Stack shape: {stack.shape}")
        print(f"Stack dtype: {stack.dtype}")
        print(f"Estimated size: {stack.nbytes / 1e9:.2f} GB")
        
        return stack, pixel_size
    
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
        
        # Add shapes layer for annotations
        # Set ndim to match the image stack (3D if stack, 2D if single image)
        if len(self.original_stack.shape) > 2:
            # 3D stack - annotations are per frame
            self.shapes_layer = self.viewer.add_shapes(
                name='Fibril Annotations',
                ndim=3,  # This ensures annotations are per-frame
                shape_type='line',  # Lines for fibril annotation
                edge_color='red',
                edge_width=2,
                face_color='transparent'
            )
        else:
            # Single 2D image
            self.shapes_layer = self.viewer.add_shapes(
                name='Fibril Annotations',
                ndim=2,
                shape_type='line',
                edge_color='red',
                edge_width=2,
                face_color='transparent'
            )
        
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
        
        # Create annotation controls widget
        @magicgui(
            call_button='Save Annotations',
            filename={'widget_type': 'FileEdit', 
                     'mode': 'w',
                     'value': 'fibril_annotations.npy',
                     'label': 'Save to:'}
        )
        def annotation_controls(filename: Path = Path('fibril_annotations.npy')):
            """Save annotations to file."""
            if self.shapes_layer is not None and len(self.shapes_layer.data) > 0:
                # Get annotation data
                annotations = {
                    'shapes': self.shapes_layer.data,
                    'shape_types': self.shapes_layer.shape_type,
                    'properties': self.shapes_layer.properties,  # Any additional properties
                    'pixel_size': self.pixel_size,
                    'mrc_files': self.mrc_files,
                    'ndim': self.shapes_layer.ndim,  # Important for reloading
                }
                
                # For 3D annotations, include frame information
                if self.shapes_layer.ndim == 3:
                    # Extract which frame each annotation belongs to
                    frame_indices = []
                    for shape in self.shapes_layer.data:
                        # First coordinate is the frame index in 3D
                        if len(shape) > 0:
                            frame_idx = int(shape[0, 0])  # z-coordinate
                            frame_indices.append(frame_idx)
                    annotations['frame_indices'] = frame_indices
                    
                    # Count annotations per frame
                    from collections import Counter
                    frame_counts = Counter(frame_indices)
                    total_frames_with_annotations = len(frame_counts)
                    
                    print(f"Annotations summary:")
                    print(f"  Total annotations: {len(self.shapes_layer.data)}")
                    print(f"  Frames with annotations: {total_frames_with_annotations}/{len(self.mrc_files)}")
                    for frame, count in sorted(frame_counts.items()):
                        if frame < len(self.mrc_files):
                            print(f"    Frame {frame} ({Path(self.mrc_files[frame]).name}): {count} annotations")
                
                # Save to file
                np.save(filename, annotations, allow_pickle=True)
                self.viewer.status = f'Saved {len(self.shapes_layer.data)} annotations to {filename}'
                print(f"\nAnnotations saved to {filename}")
            else:
                self.viewer.status = 'No annotations to save'
        
        # Add widgets to viewer
        self.viewer.window.add_dock_widget(filter_controls, area='right', name='Filter Controls')
        self.viewer.window.add_dock_widget(annotation_controls, area='right', name='Save Annotations')
        # self.viewer.window.add_dock_widget(load_annotations, area='right', name='Load Annotations') # Claude forgot to define this function
        
        # Add usage instructions
        instructions = """
        FIBRIL ANNOTATION INSTRUCTIONS:
        
        1. Use slider to adjust Butterworth lowpass filter (0 = no filter)
           - Resolution cutoff in Angstroms
           - Filter order controls sharpness (higher = sharper cutoff)
        
        2. Select 'Fibril Annotations' layer in the layer list
        
        3. Annotation tools:
           - Press 'L' for line tool (straight fibril segments)
           - Press 'Shift+L' for polyline tool (multi-segment fibrils)
           - Click to set start point, click again for end point
           - For polylines: keep clicking to add segments, double-click or Enter to finish
        
        4. Editing tools:
           - Press 'A' for selection tool to edit/delete annotations
           - Press 'D' to delete selected annotations
           - Press Escape to cancel current annotation
           - Press 'M' for pan/zoom mode
        
        5. Save/Load:
           - Use 'Save Annotations' to export your work
           - Use 'Load Annotations' to restore previous work -> NOT YET IMPLEMENTED! #TODO
        
        Navigation:
        - Scroll: zoom in/out
        - Click+drag: pan (when in pan mode 'M')
        - Slider at bottom: navigate between micrographs
        - Arrow keys: fine navigation between frames
        
        Note: Each micrograph has its own independent annotations
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
    parser.add_argument('--glob_pattern', default="*.mrc", help='Search pattern (default: "*.mrc")')

    

    parser.add_argument('--pixel-size', type=float, default=None,
                       help='Pixel size in Angstroms (if not in MRC header)')
    parser.add_argument('--permissive', action='store_true',
                       help='Force permissive mode for corrupted MRC files (use with caution)')
    
    args = parser.parse_args()
    mic_dir = Path(args.mic_dir)
    assert mic_dir.exists()
    assert mic_dir.is_dir()
    
    # Get list of micrograph mrc files:
    mrc_files = [f for f in mic_dir.glob(args.glob_pattern)]
    
    if not mrc_files:
        print("Error: No MRC files found!")
        sys.exit(1)
    
    # Check files exist
    mrc_files = [f for f in mrc_files if Path(f).exists()]
    if not mrc_files:
        print("Error: No valid MRC files found!")
        sys.exit(1)
    
    print(f"Found {len(mrc_files)} MRC files")
    
    # Create and run annotator
    annotator = CryoEMFibrilAnnotator(mrc_files, pixel_size=args.pixel_size)
    
    # Set permissive mode if requested
    if args.permissive:
        print("Warning: Running in permissive mode - corrupted files may cause issues")
        annotator.force_permissive = True
    else:
        annotator.force_permissive = False
    
    annotator.run()


if __name__ == '__main__':
    main()