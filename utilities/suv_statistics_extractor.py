from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import binary_erosion
from scipy.stats import skew, kurtosis, entropy

def advanced_prior_extractor(pet_path: Path, mask_path: Path) -> dict | None:
    """
    Estrae statistiche avanzate del primo ordine e metriche volumetriche 
    dalla maschera epatica su immagini PET.
    """
    try:
        pet_img = nib.load(pet_path)
        pet_data = pet_img.get_fdata()
        mask_data = nib.load(mask_path).get_fdata()
        
        # Recupero i voxel dimensions per il calcolo del volume
        voxel_volume_ml = np.prod(pet_img.header.get_zooms()[:3]) / 1000.0

        binary_mask = mask_data > 0
        eroded_mask = binary_erosion(binary_mask, iterations=2)
        
        suv_liver = pet_data[eroded_mask]
        suv_liver = suv_liver[suv_liver > 0]

        if len(suv_liver) == 0:
            print(f"[WARNING] Maschera vuota dopo l'erosione: {pet_path}")
            return None

        # 1. Statistiche base
        suv_mean = np.mean(suv_liver)
        suv_sd = np.std(suv_liver)
        suv_median = np.median(suv_liver)
        
        # 2. Percentili e approssimazione SUVpeak (media del top 1%)
        p75 = np.percentile(suv_liver, 75)
        p90 = np.percentile(suv_liver, 90)
        p95 = np.percentile(suv_liver, 95)
        
        top_1_percent = suv_liver[suv_liver > np.percentile(suv_liver, 99)]
        suv_peak_approx = np.mean(top_1_percent) if len(top_1_percent) > 0 else np.max(suv_liver)

        # 3. Analisi della distribuzione (First-order Radiomics)
        suv_skewness = skew(suv_liver)
        suv_kurtosis = kurtosis(suv_liver)
        
        # Entropia (approssimata su un istogramma a 100 bin)
        hist, _ = np.histogram(suv_liver, bins=100, density=True)
        hist = hist[hist > 0] # Evita log(0)
        suv_entropy = entropy(hist)

        # 4. Volume epatico eroso (utile proxy anatomico)
        liver_volume_ml = len(suv_liver) * voxel_volume_ml

        return {
            "SUV_Mean": round(suv_mean, 3),
            "SUV_Median": round(suv_median, 3),
            "SUV_SD": round(suv_sd, 3),
            "SUV_P75": round(p75, 3),
            "SUV_P90": round(p90, 3),
            "SUV_P95": round(p95, 3),
            "SUV_Peak_Approx": round(suv_peak_approx, 3),
            "Skewness": round(float(suv_skewness), 3),
            "Kurtosis": round(float(suv_kurtosis), 3),
            "Entropy": round(float(suv_entropy), 3),
            "Liver_Volume_mL": round(liver_volume_ml, 2)
        }

    except Exception as e:
        print(f"[ERROR] Fallimento nel processing di {pet_path}: {e}")
        return None

# =============================================================================
# MAIN — Dataset processing
# =============================================================================

DATASET_DIR = Path("../data/Lymphoma")
OUTPUT_CSV = Path("../data/Lymphoma_segmentation/Advanced_Prior_Stats.csv")
SUV_FILENAME = "SUV.nii.gz"
LIVER_FILENAME = "liver.nii.gz"

suv_files = sorted(DATASET_DIR.rglob(SUV_FILENAME))
results = []

for suv_path in suv_files:
    relative_parts = suv_path.relative_to(DATASET_DIR).parts
    scan_id = relative_parts[0]
    session = relative_parts[1] if len(relative_parts) > 1 else ""
    full_id = f"{scan_id}_{session}".replace(" ", "_")

    mask_path = suv_path.parent / LIVER_FILENAME
    if not mask_path.exists():
        continue

    stats = advanced_prior_extractor(suv_path, mask_path)
    if stats:
        stats["Scan_ID"] = full_id
        stats["Session"] = session
        results.append(stats)

if results:
    # Riordino le colonne per avere ID e Session all'inizio
    cols = ["Scan_ID", "Session"] + [k for k in results[0].keys() if k not in ["Scan_ID", "Session"]]
    df = pd.DataFrame(results)[cols]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Estrazione completata. Salvati {len(df)} record in {OUTPUT_CSV}")
else:
    print("Nessun risultato da salvare.")