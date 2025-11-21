#!/usr/bin/env python
'''
Convert from fibril annotations saved in .npy files to RELION-compatible .star files.
Converts line/path annotations to start-end filament coordinates for helical processing.
Set inter-box distance either absolute in Angstrom or relative by also providing the fibril diameter.
'''

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import mrcfile


def get_micrograph_dimensions(mrc_path: Path) -> Tuple[int, int]:
    """
    Get dimensions (height, width) of an MRC file.

    Args:
        mrc_path: Path to the MRC file

    Returns:
        Tuple of (height, width) in pixels
    """
    try:
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            # MRC data shape is typically (height, width) for 2D or (nz, ny, nx) for 3D
            shape = mrc.data.shape
            if len(shape) == 2:
                height, width = shape
            elif len(shape) == 3:
                # For stacks, use the first frame dimensions
                _, height, width = shape
            else:
                raise ValueError(f"Unexpected MRC dimensions: {shape}")
            return height, width
    except Exception as e:
        raise ValueError(f"Failed to read MRC file {mrc_path}: {e}")


def load_annotation_file(npy_path: Path) -> Dict[str, Any]:
    """
    Load annotation data from .npy file.

    Args:
        npy_path: Path to the .npy annotation file

    Returns:
        Dictionary containing annotation data with keys:
        - shapes: List of shape coordinates
        - shape_types: Type of shapes (line, path, etc.)
        - mrc_files: List of MRC filenames
        - pixel_size: Pixel size in Angstroms
        - frame_indices: Frame indices for 3D annotations (optional)
    """
    if not npy_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {npy_path}")

    try:
        annotations = np.load(npy_path, allow_pickle=True).item()
        print(f"Loaded annotations from {npy_path}")
        print(f"  - Number of shapes: {len(annotations.get('shapes', []))}")
        print(f"  - Shape types: {annotations.get('shape_types', 'unknown')}")
        print(f"  - Pixel size: {annotations.get('pixel_size', 'unknown')} Å/px")
        print(f"  - MRC files: {len(annotations.get('mrc_files', []))}")
        return annotations
    except Exception as e:
        raise ValueError(f"Failed to load annotation file {npy_path}: {e}")


def split_path_into_segments(shape: np.ndarray, ndim: int = 2) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split a multi-point path into individual start-end segments.

    Args:
        shape: Shape coordinate array with multiple points
        ndim: Number of dimensions (2 or 3)

    Returns:
        List of tuples (start_coords, end_coords) for each segment
    """
    segments = []

    for j in range(len(shape) - 1):
        if ndim == 3:
            start_coords = shape[j, 1:]  # [y, x]
            end_coords = shape[j+1, 1:]  # [y, x]
        else:
            start_coords = shape[j]      # [y, x]
            end_coords = shape[j+1]      # [y, x]

        segments.append((start_coords, end_coords))

    return segments


def extract_filament_coordinates(shapes: List[np.ndarray],
                                shape_types: Any,
                                frame_indices: List[int] = None,
                                ndim: int = 2,
                                split_paths: bool = False,
                                micrograph_heights: List[int] = None) -> List[Dict[str, Any]]:
    """
    Extract start-end coordinates from line/path annotations.

    Args:
        shapes: List of shape coordinate arrays
        shape_types: Shape type(s) - can be single string or list
        frame_indices: Frame indices for 3D annotations
        ndim: Number of dimensions (2 or 3)
        split_paths: If True, split multi-point paths into individual segments
        micrograph_heights: List of micrograph heights for coordinate conversion (napari to RELION)

    Returns:
        List of dictionaries with filament coordinate data
    """
    filaments = []

    # Handle single shape type or list of shape types
    if isinstance(shape_types, str):
        shape_type_list = [shape_types] * len(shapes)
    else:
        shape_type_list = shape_types

    filament_id = 1
    for i, shape in enumerate(shapes):
        if len(shape) < 2:
            print(f"Warning: Shape {i} has fewer than 2 points, skipping")
            continue

        shape_type = shape_type_list[i] if i < len(shape_type_list) else 'line'

        # Get frame index
        if ndim == 3:
            if shape.shape[1] != 3:
                print(f"Warning: Expected 3D coordinates for shape {i}, got {shape.shape[1]}D")
                continue
            frame_idx = frame_indices[i] if frame_indices else int(shape[0, 0])
        else:
            if shape.shape[1] != 2:
                print(f"Warning: Expected 2D coordinates for shape {i}, got {shape.shape[1]}D")
                continue
            frame_idx = 0  # Single frame

        # Get micrograph height for coordinate conversion
        if micrograph_heights and frame_idx < len(micrograph_heights):
            micrograph_height = micrograph_heights[frame_idx]
        else:
            micrograph_height = None

        # Split multi-point paths into segments if requested
        if split_paths and len(shape) > 2:
            segments = split_path_into_segments(shape, ndim)
            for start_coords, end_coords in segments:
                # Convert napari coordinates (top-left origin) to RELION (bottom-left origin)
                start_y_napari = float(start_coords[0])
                end_y_napari = float(end_coords[0])

                if micrograph_height is not None:
                    start_y = micrograph_height - start_y_napari
                    end_y = micrograph_height - end_y_napari
                else:
                    start_y = start_y_napari
                    end_y = end_y_napari

                filaments.append({
                    'filament_id': filament_id,
                    'frame_index': frame_idx,
                    'start_x': float(start_coords[1]),  # x coordinate
                    'start_y': start_y,  # y coordinate (converted)
                    'end_x': float(end_coords[1]),      # x coordinate
                    'end_y': end_y,      # y coordinate (converted)
                    'shape_type': shape_type,
                    'length_pixels': float(np.linalg.norm(end_coords - start_coords))
                })
                filament_id += 1
        else:
            # Use first and last point only
            if ndim == 3:
                start_coords = shape[0, 1:]  # [y, x]
                end_coords = shape[-1, 1:]   # [y, x]
            else:
                start_coords = shape[0]   # [y, x]
                end_coords = shape[-1]    # [y, x]

            # Convert napari coordinates (top-left origin) to RELION (bottom-left origin)
            start_y_napari = float(start_coords[0])
            end_y_napari = float(end_coords[0])

            if micrograph_height is not None:
                start_y = micrograph_height - start_y_napari
                end_y = micrograph_height - end_y_napari
            else:
                start_y = start_y_napari
                end_y = end_y_napari

            filaments.append({
                'filament_id': filament_id,
                'frame_index': frame_idx,
                'start_x': float(start_coords[1]),  # x coordinate
                'start_y': start_y,  # y coordinate (converted)
                'end_x': float(end_coords[1]),      # x coordinate
                'end_y': end_y,      # y coordinate (converted)
                'shape_type': shape_type,
                'length_pixels': float(np.linalg.norm(end_coords - start_coords))
            })
            filament_id += 1

    return filaments


def generate_particle_coordinates(filaments: List[Dict[str, Any]],
                                 inter_box_distance: float,
                                 pixel_size: float,
                                 box_size: int = 256) -> List[Dict[str, Any]]:
    """
    Generate particle coordinates along filaments at specified intervals.

    Args:
        filaments: List of filament coordinate dictionaries
        inter_box_distance: Distance between particle boxes in Angstroms
        pixel_size: Pixel size in Angstroms per pixel
        box_size: Box size for particle extraction in pixels

    Returns:
        List of particle coordinate dictionaries
    """
    particles = []
    particle_id = 1

    # Convert inter-box distance from Angstroms to pixels
    inter_box_pixels = inter_box_distance / pixel_size

    for filament in filaments:
        start_x, start_y = filament['start_x'], filament['start_y']
        end_x, end_y = filament['end_x'], filament['end_y']

        # Calculate filament vector and length
        dx = end_x - start_x
        dy = end_y - start_y
        length_pixels = np.sqrt(dx**2 + dy**2)

        if length_pixels < inter_box_pixels:
            print(f"Warning: Filament {filament['filament_id']} too short for particle extraction")
            continue

        # Number of particles that can fit along this filament
        num_particles = int(length_pixels / inter_box_pixels)

        # Unit vector along filament
        ux = dx / length_pixels
        uy = dy / length_pixels

        for i in range(num_particles):
            # Position along filament
            distance = i * inter_box_pixels

            # Particle coordinates
            particle_x = start_x + distance * ux
            particle_y = start_y + distance * uy

            # Calculate psi angle (filament direction angle in degrees)
            psi_angle = np.degrees(np.arctan2(dy, dx))

            particles.append({
                'particle_id': particle_id,
                'filament_id': filament['filament_id'],
                'frame_index': filament['frame_index'],
                'coordinate_x': float(particle_x),
                'coordinate_y': float(particle_y),
                'psi_angle': float(psi_angle),
                'distance_along_filament': float(distance * pixel_size)  # in Angstroms
            })

            particle_id += 1

    return particles


def create_manualpick_star_files(filaments: List[Dict[str, Any]],
                                 mrc_files: List[str],
                                 output_dir: Path) -> None:
    """
    Create per-micrograph RELION manual pick .star files with start-end coordinates.

    Args:
        filaments: List of filament coordinate dictionaries
        mrc_files: List of MRC filenames
        output_dir: Output directory for .star files
    """
    if not filaments:
        raise ValueError("No filaments to write to .star files")

    # Group filaments by frame/micrograph
    filaments_by_frame = {}
    for filament in filaments:
        frame_idx = filament['frame_index']
        if frame_idx not in filaments_by_frame:
            filaments_by_frame[frame_idx] = []
        filaments_by_frame[frame_idx].append(filament)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create one .star file per micrograph
    for frame_idx in sorted(filaments_by_frame.keys()):
        if frame_idx < len(mrc_files):
            micrograph_path = Path(mrc_files[frame_idx])
            micrograph_stem = micrograph_path.stem
        else:
            micrograph_stem = f"frame_{frame_idx:04d}"

        output_file = output_dir / f"{micrograph_stem}_manualpick.star"
        frame_filaments = filaments_by_frame[frame_idx]

        # Create STAR file content
        star_lines = []
        star_lines.append("")
        star_lines.append("# version 30001")
        star_lines.append("")
        star_lines.append("data_")
        star_lines.append("")
        star_lines.append("loop_ ")
        star_lines.append("_rlnCoordinateX #1 ")
        star_lines.append("_rlnCoordinateY #2 ")
        star_lines.append("_rlnParticleSelectionType #3 ")
        star_lines.append("_rlnAnglePsi #4 ")
        star_lines.append("_rlnAutopickFigureOfMerit #5 ")

        # Write filament start-end coordinates
        for filament in frame_filaments:
            # Start coordinate
            start_x = filament['start_x']
            start_y = filament['start_y']
            star_lines.append(f"{start_x:12.6f} {start_y:12.6f}            2   -999.00000   -999.00000 ")

            # End coordinate
            end_x = filament['end_x']
            end_y = filament['end_y']
            star_lines.append(f"{end_x:12.6f} {end_y:12.6f}            2   -999.00000   -999.00000 ")

        star_lines.append("")

        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(star_lines))

        print(f"Created {output_file.name}: {len(frame_filaments)} filaments ({len(frame_filaments)*2} coordinates)")


def create_star_file(particles: List[Dict[str, Any]],
                    mrc_files: List[str],
                    pixel_size: float,
                    output_path: Path) -> None:
    """
    Create RELION .star file from particle coordinates.

    Args:
        particles: List of particle coordinate dictionaries
        mrc_files: List of MRC filenames
        pixel_size: Pixel size in Angstroms per pixel
        output_path: Output .star file path
    """
    if not particles:
        raise ValueError("No particles to write to .star file")

    # Group particles by frame/micrograph
    particles_by_frame = {}
    for particle in particles:
        frame_idx = particle['frame_index']
        if frame_idx not in particles_by_frame:
            particles_by_frame[frame_idx] = []
        particles_by_frame[frame_idx].append(particle)

    # Create STAR file content
    star_lines = []
    star_lines.append("# RELION format star file")
    star_lines.append("# Created by convert_to_star.py")
    star_lines.append("")
    star_lines.append("data_")
    star_lines.append("")
    star_lines.append("loop_")
    star_lines.append("_rlnCoordinateX #1")
    star_lines.append("_rlnCoordinateY #2")
    star_lines.append("_rlnMicrographName #3")
    star_lines.append("_rlnAnglePsi #4")
    star_lines.append("_rlnHelicalTrackLength #5")
    star_lines.append("_rlnHelicalTubeID #6")

    # Write particle data
    for frame_idx in sorted(particles_by_frame.keys()):
        if frame_idx < len(mrc_files):
            micrograph_name = mrc_files[frame_idx]
        else:
            micrograph_name = f"frame_{frame_idx:04d}.mrc"

        frame_particles = particles_by_frame[frame_idx]

        for particle in frame_particles:
            # Convert coordinates to RELION format (x, y in pixels)
            coord_x = particle['coordinate_x']
            coord_y = particle['coordinate_y']
            psi_angle = particle['psi_angle']
            tube_id = particle['filament_id']
            track_length = particle['distance_along_filament']

            star_lines.append(f"{coord_x:8.2f} {coord_y:8.2f} {micrograph_name} {psi_angle:8.2f} {track_length:8.2f} {tube_id}")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(star_lines))

    print(f"Created .star file: {output_path}")
    print(f"  - Total particles: {len(particles)}")
    print(f"  - Micrographs: {len(particles_by_frame)}")
    print(f"  - Pixel size: {pixel_size} Å/px")


def main():
    """Main function to handle command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert fibril annotations from .npy to RELION .star format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion with micrograph shape (RECOMMENDED - fast)
  python convert_to_star.py annotations.npy --mic_shape 4096 4096   # For Falcon 4 camera
  python convert_to_star.py annotations.npy --mic_shape 4092 5760   # For Gatan K3 (non-superresolution)

  # Override pixel size if MRC header has incorrect value
  python convert_to_star.py annotations.npy --mic_shape 4096 4096 --pixel_size 1.05

  # Export manual pick files with micrograph shape
  python convert_to_star.py annotations.npy --manualpick --mic_shape 4096 4096

  # Custom inter-box distance with coordinate conversion
  python convert_to_star.py annotations.npy --inter_box_distance 150 --mic_shape 4096 4096

  # Export manual pick files and split multi-point paths into segments
  python convert_to_star.py annotations.npy --manualpick --split_paths --mic_shape 4096 4096

  # Alternative: Load MRC files to determine shape automatically (slower)
  python convert_to_star.py annotations.npy --manualpick --mrc_dir /path/to/micrographs/

  # Specify output file
  python convert_to_star.py annotations.npy -o fibrils.star --mic_shape 4096 4096

Note: Either --mic_shape or --mrc_dir is required to correctly convert y-coordinates from
napari (top-left origin) to RELION (bottom-left origin) convention. Without either option,
coordinates will appear vertically mirrored in RELION. Use --mic_shape for speed when all
micrographs have the same dimensions.
        """
    )

    parser.add_argument('npy_file', type=Path,
                       help='Input .npy annotation file')
    parser.add_argument('-o', '--output', type=Path,
                       help='Output .star file (default: based on input filename)')
    parser.add_argument('--inter_box_distance', type=float, default=100.0,
                       help='Inter-box distance in Angstroms (default: 100)')
    parser.add_argument('--box_size', type=int, default=256,
                       help='Box size for particle extraction in pixels (default: 256)')
    parser.add_argument('--manualpick', action='store_true',
                       help='Export per-micrograph manual pick star files instead of particle coordinates')
    parser.add_argument('--manualpick_dir', type=Path,
                       help='Output directory for manual pick star files (default: {input}_manualpick/)')
    parser.add_argument('--split_paths', action='store_true',
                       help='Split multi-point paths into individual start-end segments')
    parser.add_argument('--mrc_dir', type=Path,
                       help='Directory containing MRC files for coordinate conversion (required for correct y-axis)')
    parser.add_argument('--mic_shape', type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'),
                       help='Micrograph shape as HEIGHT WIDTH (alternative to --mrc_dir, faster)')
    parser.add_argument('--pixel_size', type=float,
                       help='Override pixel size in Angstroms (use if MRC header pixel size is incorrect)')

    args = parser.parse_args()

    # Validate input file
    if not args.npy_file.exists():
        print(f"Error: Input file not found: {args.npy_file}")
        return 1

    if not args.npy_file.suffix == '.npy':
        print(f"Error: Input file must have .npy extension: {args.npy_file}")
        return 1

    # Set output file
    if args.output:
        output_file = args.output
    else:
        output_file = args.npy_file.with_suffix('.star')

    try:
        # Load annotation file
        print(f"Loading annotations from {args.npy_file}")
        annotations = load_annotation_file(args.npy_file)

        # Handle pixel size override
        annotation_pixel_size = annotations.get('pixel_size', 1.0)

        if args.pixel_size:
            # User specified pixel size - use it
            pixel_size = args.pixel_size
            print(f"Using user-specified pixel size: {pixel_size} Å/px (annotation file has {annotation_pixel_size} Å/px)")
        else:
            # Use pixel size from annotation file
            pixel_size = annotation_pixel_size

            # Warn if pixel size looks suspicious (exactly 1.0 Å is often incorrect)
            if abs(pixel_size - 1.0) < 0.001:
                print("=" * 70)
                print("WARNING: Pixel size is 1.0 Å/px, which is likely incorrect!")
                print("This may indicate missing or wrong pixel size in the MRC header.")
                print("Consider using --pixel_size to specify the correct value.")
                print("=" * 70)

        # Get micrograph dimensions for coordinate conversion
        micrograph_heights = None

        if args.mic_shape:
            # Use provided micrograph shape (fast method)
            height, width = args.mic_shape
            num_micrographs = len(annotations.get('mrc_files', []))
            micrograph_heights = [height] * num_micrographs
            print(f"Using provided micrograph shape: {width}x{height} pixels")
            print(f"Applying to {num_micrographs} micrographs")

        elif args.mrc_dir:
            # Load MRC files to get micrograph dimensions (slow but accurate)
            print(f"Loading MRC files from {args.mrc_dir} for coordinate conversion...")
            mrc_files = annotations.get('mrc_files', [])
            micrograph_heights = []

            for mrc_filename in mrc_files:
                mrc_path = args.mrc_dir / mrc_filename
                if mrc_path.exists():
                    try:
                        height, width = get_micrograph_dimensions(mrc_path)
                        micrograph_heights.append(height)
                        print(f"  {mrc_filename}: {width}x{height} pixels")
                    except Exception as e:
                        print(f"  Warning: Failed to read {mrc_filename}: {e}")
                        micrograph_heights.append(None)
                else:
                    print(f"  Warning: MRC file not found: {mrc_path}")
                    micrograph_heights.append(None)

            if not micrograph_heights:
                print("Warning: No MRC files found. Coordinates will not be converted.")
                micrograph_heights = None
        else:
            print("Warning: Neither --mic_shape nor --mrc_dir specified.")
            print("         Y-coordinates will NOT be converted to RELION convention.")
            print("         This may cause coordinates to appear mirrored in RELION.")
            micrograph_heights = None

        # Extract filament coordinates
        print("Extracting filament coordinates...")
        filaments = extract_filament_coordinates(
            annotations['shapes'],
            annotations.get('shape_types', 'line'),
            annotations.get('frame_indices'),
            annotations.get('ndim', 2),
            split_paths=args.split_paths,
            micrograph_heights=micrograph_heights
        )

        if not filaments:
            print("Error: No valid filaments found in annotation file")
            return 1

        print(f"Found {len(filaments)} filaments")

        # Export based on mode
        if args.manualpick:
            # Set output directory for manual pick files
            if args.manualpick_dir:
                output_dir = args.manualpick_dir
            else:
                output_dir = args.npy_file.parent / f"{args.npy_file.stem}_manualpick"

            # Create per-micrograph manual pick star files
            print(f"Creating per-micrograph manual pick .star files...")
            create_manualpick_star_files(
                filaments,
                annotations.get('mrc_files', []),
                output_dir
            )

            print(f"\nConversion completed successfully!")
            print(f"Input:  {args.npy_file}")
            print(f"Output: {output_dir}/")
            print(f"  - Total filaments/lines: {len(filaments)}")
            if args.split_paths:
                print(f"  - Multi-point paths split into segments")
        else:
            # Generate particle coordinates
            print(f"Generating particle coordinates with {args.inter_box_distance} Å spacing...")
            particles = generate_particle_coordinates(
                filaments,
                args.inter_box_distance,
                pixel_size,
                args.box_size
            )

            if not particles:
                print("Error: No particles generated from filaments")
                return 1

            # Create STAR file
            print("Creating RELION .star file...")
            create_star_file(
                particles,
                annotations.get('mrc_files', []),
                pixel_size,
                output_file
            )

            print(f"\nConversion completed successfully!")
            print(f"Input:  {args.npy_file}")
            print(f"Output: {output_file}")

        return 0

    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


if __name__ == "__main__":
    exit(main())