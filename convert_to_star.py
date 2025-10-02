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


def extract_filament_coordinates(shapes: List[np.ndarray],
                                shape_types: Any,
                                frame_indices: List[int] = None,
                                ndim: int = 2) -> List[Dict[str, Any]]:
    """
    Extract start-end coordinates from line/path annotations.

    Args:
        shapes: List of shape coordinate arrays
        shape_types: Shape type(s) - can be single string or list
        frame_indices: Frame indices for 3D annotations
        ndim: Number of dimensions (2 or 3)

    Returns:
        List of dictionaries with filament coordinate data
    """
    filaments = []

    # Handle single shape type or list of shape types
    if isinstance(shape_types, str):
        shape_type_list = [shape_types] * len(shapes)
    else:
        shape_type_list = shape_types

    for i, shape in enumerate(shapes):
        if len(shape) < 2:
            print(f"Warning: Shape {i} has fewer than 2 points, skipping")
            continue

        shape_type = shape_type_list[i] if i < len(shape_type_list) else 'line'

        # For 3D annotations, coordinates are [frame, y, x]
        # For 2D annotations, coordinates are [y, x]
        if ndim == 3:
            if shape.shape[1] != 3:
                print(f"Warning: Expected 3D coordinates for shape {i}, got {shape.shape[1]}D")
                continue
            frame_idx = frame_indices[i] if frame_indices else int(shape[0, 0])
            start_coords = shape[0, 1:]  # [y, x]
            end_coords = shape[-1, 1:]   # [y, x]
        else:
            if shape.shape[1] != 2:
                print(f"Warning: Expected 2D coordinates for shape {i}, got {shape.shape[1]}D")
                continue
            frame_idx = 0  # Single frame
            start_coords = shape[0]   # [y, x]
            end_coords = shape[-1]    # [y, x]

        filaments.append({
            'filament_id': i + 1,
            'frame_index': frame_idx,
            'start_x': float(start_coords[1]),  # x coordinate
            'start_y': float(start_coords[0]),  # y coordinate
            'end_x': float(end_coords[1]),      # x coordinate
            'end_y': float(end_coords[0]),      # y coordinate
            'shape_type': shape_type,
            'length_pixels': float(np.linalg.norm(end_coords - start_coords))
        })

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
  # Basic conversion with default 100 Å spacing
  python convert_to_star.py annotations.npy

  # Custom inter-box distance
  python convert_to_star.py annotations.npy --inter_box_distance 150

  # Specify output file
  python convert_to_star.py annotations.npy -o fibrils.star

  # Custom box size for extraction
  python convert_to_star.py annotations.npy --box_size 512
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

        # Extract filament coordinates
        print("Extracting filament coordinates...")
        filaments = extract_filament_coordinates(
            annotations['shapes'],
            annotations.get('shape_types', 'line'),
            annotations.get('frame_indices'),
            annotations.get('ndim', 2)
        )

        if not filaments:
            print("Error: No valid filaments found in annotation file")
            return 1

        print(f"Found {len(filaments)} filaments")

        # Generate particle coordinates
        print(f"Generating particle coordinates with {args.inter_box_distance} Å spacing...")
        particles = generate_particle_coordinates(
            filaments,
            args.inter_box_distance,
            annotations['pixel_size'],
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
            annotations['pixel_size'],
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