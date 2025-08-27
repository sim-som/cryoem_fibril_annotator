import os
from pathlib import Path
from typing import Optional, Tuple, List

def find_corresponding_file(
    input_file: str, 
    target_folder: str, 
    micrograph_suffixes: List[str] = None,
    power_spectrum_suffixes: List[str] = None
) -> Optional[str]:
    """
    Find the corresponding cryo-EM data file between micrographs and power spectra.
    
    Args:
        input_file (str): Path to the input file (micrograph or power spectrum)
        target_folder (str): Path to folder containing the corresponding files
        micrograph_suffixes (List[str]): List of possible micrograph suffixes
        power_spectrum_suffixes (List[str]): List of possible power spectrum suffixes
    
    Returns:
        Optional[str]: Path to corresponding file, or None if not found
    
    Example:
        # Find power spectrum corresponding to a micrograph
        micrograph = "path/to/FoilHole_11886543_Data_11866787_53_20250612_234250_fractions_patch_aligned_doseweighted.mrc"
        power_spectrum_folder = "path/to/power_spectra/"
        corresponding = find_corresponding_file(micrograph, power_spectrum_folder)
    """
    
    # Default suffixes commonly used in cryo-EM processing
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
        print(f"Warning: Target folder {target_folder} does not exist")
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
        print(f"Warning: Could not identify file type for {input_name}")
        return None
    
    # Search for corresponding file with any of the target suffixes
    for suffix in target_suffixes:
        candidate_file = target_path / (base_name + suffix)
        if candidate_file.exists():
            return str(candidate_file)
    
    print(f"Warning: No corresponding file found for {input_name} in {target_folder}")
    return None


def find_file_pairs(
    folder1: str, 
    folder2: str,
    micrograph_suffixes: List[str] = None,
    power_spectrum_suffixes: List[str] = None
) -> List[Tuple[str, str]]:
    """
    Find all corresponding file pairs between two folders.
    
    Args:
        folder1 (str): Path to first folder
        folder2 (str): Path to second folder
        micrograph_suffixes (List[str]): List of possible micrograph suffixes
        power_spectrum_suffixes (List[str]): List of possible power spectrum suffixes
    
    Returns:
        List[Tuple[str, str]]: List of tuples containing paired file paths
    """
    
    folder1_path = Path(folder1)
    folder2_path = Path(folder2)
    
    if not folder1_path.exists() or not folder2_path.exists():
        print("Warning: One or both folders do not exist")
        return []
    
    pairs = []
    
    # Check all files in folder1 for matches in folder2
    for file1 in folder1_path.iterdir():
        if file1.is_file():
            corresponding = find_corresponding_file(
                str(file1), 
                str(folder2_path),
                micrograph_suffixes,
                power_spectrum_suffixes
            )
            if corresponding:
                pairs.append((str(file1), corresponding))
    
    return pairs


def validate_correspondence(file1: str, file2: str) -> bool:
    """
    Validate that two files are indeed corresponding based on their base names.
    
    Args:
        file1 (str): Path to first file
        file2 (str): Path to second file
    
    Returns:
        bool: True if files correspond, False otherwise
    """
    
    # Extract base names for comparison
    micrograph_suffixes = [
        '_doseweighted.mrc', '_aligned.mrc', '_motion_corrected.mrc',
        '_patch_aligned_doseweighted.mrc', '_fractions_patch_aligned_doseweighted.mrc'
    ]
    power_spectrum_suffixes = [
        '_ctf_diag_2D.mrc', '_ctf_2D.mrc', '_power_spectrum.mrc',
        '_ps.mrc', '_ctf_diag.mrc'
    ]
    
    name1 = Path(file1).name
    name2 = Path(file2).name
    
    base1 = base2 = None
    
    # Extract base name from file1
    for suffix in micrograph_suffixes + power_spectrum_suffixes:
        if name1.endswith(suffix):
            base1 = name1[:-len(suffix)]
            break
    
    # Extract base name from file2  
    for suffix in micrograph_suffixes + power_spectrum_suffixes:
        if name2.endswith(suffix):
            base2 = name2[:-len(suffix)]
            break
    
    return base1 is not None and base1 == base2


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    micrograph_file = "FoilHole_11886543_Data_11866787_53_20250612_234250_fractions_patch_aligned_doseweighted.mrc"
    power_spectrum_file = "FoilHole_11886543_Data_11866787_53_20250612_234250_fractions_patch_aligned_ctf_diag_2D.mrc"
    
    # Test validation
    print("Testing file correspondence validation:")
    print(f'Micrograph file: "FoilHole_11886543_Data_11866787_53_20250612_234250_fractions_patch_aligned_doseweighted.mrc"')
    print(f'2D Power spectrum file: "FoilHole_11886543_Data_11866787_53_20250612_234250_fractions_patch_aligned_ctf_diag_2D.mrc"')
    print(f"Files correspond: {validate_correspondence(micrograph_file, power_spectrum_file)}")
    
    # Example of how to use the function
    ## Find a micrographs corresponding power spectrum: 
    print("\nExample usage:")
    print("# Given a micrograph, find its power spectrum")
    print("micrograph = '/path/to/micrographs/file.mrc'")
    print("power_spectra_folder = '/path/to/power_spectra/'")
    print("corresponding = find_corresponding_file(micrograph, power_spectra_folder)")

    micrograph = Path("/home/simon/jureca_erc_CS_mount/jobs/SiSo/CS-aros-mutation/S1/motioncorrected/FoilHole_11890730_Data_11897769_19_20250613_040626_fractions_patch_aligned_doseweighted.mrc")
    assert micrograph.exists()
    ctf_folder = "/home/simon/jureca_erc_CS_mount/jobs/SiSo/CS-aros-mutation/S1/ctfestimated"
    ps_file = find_corresponding_file(micrograph, ctf_folder)
    assert Path(ps_file).exists()
    print(ps_file)
    
    ## Find all pairs between two folders:
    print("\n# Find all pairs between two folders")
    print("pairs = find_file_pairs('/path/to/micrographs/', '/path/to/power_spectra/')")
    print("for mic, ps in pairs:")
    print("    print(f'Micrograph: {mic} <-> Power Spectrum: {ps}')")

    mic_folder = Path("/home/simon/jureca_erc_CS_mount/jobs/SiSo/CS-aros-mutation/S1/motioncorrected")

    pairs = find_file_pairs(mic_folder, ctf_folder)
    for mic, ps in pairs:
        print(f'Micrograph: {mic} <-> Power Spectrum: {ps}')
    