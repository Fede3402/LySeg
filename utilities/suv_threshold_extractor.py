from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import binary_erosion


def threshold_extractor(pet_path: Path, mask_path: Path) -> dict | None:
    """
    Load a PET image and its corresponding liver mask,
    then compute SUV statistics and dynamic thresholds.

    Parameters
    ----------
    pet_path  : Path  Path to the PET NIfTI file (.nii / .nii.gz)
    mask_path : Path  Path to the liver mask NIfTI file (.nii / .nii.gz)

    Returns
    -------
    dict with SUV statistics and thresholds, or None on failure.
    """
    try:
        # 1. Load 3-D tensors
        pet_data  = nib.load(pet_path).get_fdata()
        mask_data = nib.load(mask_path).get_fdata()

        
        # 2. Binarise and erode
        # Remove 2 voxels of border to avoid contamination from adjacent organs
        binary_mask = mask_data > 0
        eroded_mask = binary_erosion(binary_mask, iterations=2)
        

        # 3. Extract SUV values inside the eroded mask
        suv_liver = pet_data[eroded_mask]

        # Keep only strictly positive values
        suv_liver = suv_liver[suv_liver > 0]

        # Guard: empty mask after erosion
        if len(suv_liver) == 0:
            print(f"  [WARNING] Empty mask after erosion for: {pet_path}")
            return None

        # 4. Compute descriptive statistics
        suv_mean   = np.mean(suv_liver)
        suv_sd     = np.std(suv_liver)
        suv_median = np.median(suv_liver)

        # 5. Compute dynamic thresholds
        threshold_percist    = suv_mean * 1.5            # Multiplicative (PERCIST-style)
        threshold_mean_2sd   = suv_mean + (2 * suv_sd)  # Statistical (Mean + 2 SD)

        return {
            "SUV_Mean"         : round(suv_mean,             3),
            "SUV_Median"       : round(suv_median,           3),
            "SUV_SD"           : round(suv_sd,               3),
            "Threshold_1.5x"   : round(threshold_percist,    3),
            "Threshold_Mean_2SD": round(threshold_mean_2sd,  3),
        }

    except Exception as e:
        print(f"  [ERROR] Failed to process {pet_path}: {e}")
        return None


# =============================================================================
# MAIN — Dataset processing
# =============================================================================

# Replace with the actual path to your dataset folder
DATASET_DIR = Path("../data/Lymphoma")

OUTPUT_COLUMNS = [
    "Scan_ID",
    "Session",
    "SUV_Mean",
    "SUV_Median",
    "SUV_SD",
    "Threshold_1.5x",
    "Threshold_Mean_2SD",
]

OUTPUT_CSV = Path("../data/Lymphoma_segmentation/Dynamic_Thresholds.csv")

# Fixed filenames to look for
SUV_FILENAME   = "SUV.nii.gz"
LIVER_FILENAME = "liver.nii.gz"

# Auto-discover all SUV files recursively.
#
# Expected layout:
#   DATASET_DIR/
#   └── <scan_id>/           e.g. PETCT_0beb67c923
#       └── <session>/       e.g. 07-25-1999-NA-PET-CT Ganzkoerper primaer mit KM-37911
#           ├── SUV.nii.gz
#           └── liver.nii.gz
suv_files = sorted(DATASET_DIR.rglob(SUV_FILENAME))

if not suv_files:
    print(f"[WARNING] No '{SUV_FILENAME}' files found under '{DATASET_DIR}'. Check the path.")
else:
    print(f"Found {len(suv_files)} SUV file(s). Starting extraction...\n")

results = []

for suv_path in suv_files:
    relative_parts = suv_path.relative_to(DATASET_DIR).parts

    scan_id = relative_parts[0]

    session = ""
    if len(relative_parts) > 1:
        session = relative_parts[1]

    # Full unique patient/session identifier
    full_id = f"{scan_id}_{session}"

    # Clean spaces for CSV compatibility
    full_id = full_id.replace(" ", "_")

    # The liver mask is expected to sit in the same folder as the SUV file
    mask_path = suv_path.parent / LIVER_FILENAME

    if not mask_path.exists():
        print(f"[{scan_id}] '{LIVER_FILENAME}' not found next to SUV file, skipping.")
        continue

    stats = threshold_extractor(suv_path, mask_path)

    if stats:
        stats["Scan_ID"] = full_id
        stats["Session"] = session
        results.append(stats)
        print(f"[{scan_id}]  1.5x threshold = {stats['Threshold_1.5x']} "
              f"| Mean+2SD threshold = {stats['Threshold_Mean_2SD']}")
    else:
        print(f"[{scan_id}] Invalid liver mask, skipping.")

# Save results
if results:
    df = pd.DataFrame(results)[OUTPUT_COLUMNS]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Results saved to '{OUTPUT_CSV}' ({len(df)} patient(s) processed).")
else:
    print("\nNo valid results to save.")