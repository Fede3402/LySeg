import os 
import glob 
import random 

from typing import List, Optional

import ants 


# --- PARAMETERS CONFIGURATION ---
RANDOM_SEED: int = 42
SAMPLES_PER_GROUP: int = 25
TARGET_SPACING: tuple[float, float, float] = (3.0, 3.0, 3.0)
TEMPLATE_ITERATIONS: int = 3
TEMPLATE_TRANSFORM: str = 'SyN'



def find_res_ct(patient_id : str, base_dir : str) -> Optional[str]:
    
    search_pattern = os.path.join(base_dir, patient_id,"**","CTres.nii.gz")
    ct_res_path = glob.glob(search_pattern, recursive=True)
    
    return ct_res_path[0] if ct_res_path else None

def build_atlas_template(
          negatives_dir: str,
          list_youngpz_path: str,
          list_oldpz_path: str,
          out_template_dir: str,
) -> None:
    
    print('Building anatomica template reconstruction...')


    if not os.path.exists(list_youngpz_path) or not os.path.exists(list_oldpz_path):
        raise FileNotFoundError(
            "Patient lists not found. Run 'split_metadata.py' before this script."
        )
    
    os.makedirs(out_template_dir, exist_ok=True)
    
    with open(list_youngpz_path, 'r') as f:
            ids_18_40: List[str] = [line.strip() for line in f if line.strip()]
    with open(list_oldpz_path, 'r') as f:
        ids_41_plus: List[str] = [line.strip() for line in f if line.strip()]

    random.seed(RANDOM_SEED)

    sample_young = random.sample(ids_18_40, min(SAMPLES_PER_GROUP, len(ids_18_40)))
    sample_old = random.sample(ids_41_plus, min(SAMPLES_PER_GROUP, len(ids_41_plus)))
    selected_patients : List[str] = sample_young + sample_old

    print(f" Number of selected patients: {len(selected_patients)}"
          f"({len(sample_young)} Giovani, {len(sample_old)} Adulti)")
    
    # Loading and spatial resampling 
    ct_list = []
    for patient_id in selected_patients:
        ct_path = find_res_ct(patient_id, negatives_dir)
        
        if not ct_path:
             print(f" Skipping patient : {patient_id}, CTres not found")
             continue
        
        try:
            ct = ants.image_read(ct_path)
            ct_res = ants.resample_image(
                  ct,
                  target_spacing=TARGET_SPACING,
                  use_voxels = False,
                  interp_type = 1
            )
            ct_list.append(ct_res)

            print(f" Loaded and resample patient : {patient_id}, Shape: {ct_res.shape()}")
        
        except Exception as e:
             print (f" [Error]  Not possible to process patient {patient_id} : {str(e)}")

    print(f" Starting registration ANTs with {len(ct_list)} volumes"
          f" Iterations : {TEMPLATE_ITERATIONS}, Transform : {TEMPLATE_TRANSFORM}")
    
    template = ants.build_template(
         initial_template = None,
         image_list = ct_list,
         iterations = TEMPLATE_ITERATIONS,
         transform = TEMPLATE_TRANSFORM,
         gradient_step = 0.2
    )

    out_file_path = os.path.join(out_template_dir, "CT_WholeBody_Template.nii.gz")
    ants.image_write(template, out_file_path)
    
    print(f" Generated template in file: {out_file_path}")
            

if __name__ == "__main__":

    # Dynamic path configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "data"))

    negatives_dir = os.path.join(data_dir, "Negative")
    list_youngpz_path = os.path.join(data_dir, "Metadata for atlas creation", "atlas_18_40_ids.txt")
    list_oldpz_path = os.path.join(data_dir, "Metadata for atlas creation", "atlas_41_plus_ids.txt")

    out_template_dir = os.path.join(data_dir, "Template")
   

    build_atlas_template(
         negatives_dir,
         list_youngpz_path, 
         list_oldpz_path, 
         out_template_dir
         )

