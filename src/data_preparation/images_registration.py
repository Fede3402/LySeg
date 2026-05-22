import os 
import glob 
from typing import List, Optional

import ants 


# -- CONFIGURATION PARAMETERS -- 

TARGET_SPACING: tuple[float, float, float] = (3.0, 3.0, 3.0)
REGISTRATION_TRANSFORM: str = 'SyN'

def find_patient_volumes(patient_id: str, base_dir: str) -> Optional[List[str]]:
    
    ct_pattern = os.path.join(base_dir, patient_id,"**","*.nii.gz")
    pet_pattern = os.path.join(base_dir, patient_id,"**","*.nii.gz")

    ct_path = glob.glob(ct_pattern, recursive=True)
    pet_path = glob.glob(pet_pattern, recursive=True)

    return ct_path, pet_path if ct_path else None

def register_cohort_to_template(
        template_path: str,
        negatives_dir: str,
        list_ids:str,
        output_dir: str
):
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found at {template_path}.")
        
    os.makedirs(output_dir, exist_ok=True)

    # Template loading
    fixed_template = ants.image_read(template_path)

    all_patients_id = []
    for list_file in list_ids:
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                ids = [line.strip() for line in f if line.strip()]
                all_patients_id.extend(ids)

    total_patients = len(all_patients_id)
    print(f" Number of patients: {total_patients}")

    # Registration loop 
    for idx, patient_id in enumerate(all_patients_id, 1):
        out_pet_path = os.path.join(output_dir, f"{patient_id}_warped_pet.nii.gz")
        out_ct_path = os.path.join(output_dir, f"{patient_id}_warped_ct.nii.gz")

        if os.path.exists(out_pet_path):
            print(f"[{idx}/{total_patients}] {patient_id} già processato. Salto...")
            continue

        print(f"[{idx}/{total_patients}] Patient registration: {patient_id}")

        ct_path, pet_path = find_patient_volumes(patient_id, negatives_dir)

        if not ct_path or not pet_path:
            print(f" Missing files for {patient_id}")
            continue

        try:

            moving_ct = ants.image_read(ct_path)
            moving_pet = ants.image_read(pet_path)

            # Resampling of ct before registration 
            moving_ct_res = ants.resample_image(
                moving_ct,
                target_spacing=TARGET_SPACING,
                use_voxels = False,
                interp_type = 1
            )

            # CT registration 
            registration = ants.registration(
                fixed = fixed_template,
                moving = moving_ct_res,
                type_of_transform = REGISTRATION_TRANSFORM,
                verbose = True
            )

            ants.image_write(registration['warpedmovout'], out_ct_path)

            # Application of deformation to PET
            warped_pet = ants.apply_transforms(
                    fixed=fixed_template,
                    moving=moving_pet,
                    transformlist=registration['fwdtransforms'],
                    interpolator='linear' 
                )
                
            ants.image_write(warped_pet, out_pet_path)
            print(f"  Imaged saved for: {patient_id}.")

        except Exception as e:
            print(f" Operation failed for: {patient_id}. Error: {str(e)}")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data"))
    
    # Input
    DIR_NEGATIVES = os.path.join(DATA_DIR, "Negative")
    FILE_TEMPLATE = os.path.join(DATA_DIR, "templates", "CT_WholeBody_Template.nii.gz")
    
    LISTS = [
        os.path.join(DATA_DIR, "patient_lists", "atlas_18_40_ids.txt"),
        os.path.join(DATA_DIR, "patient_lists", "atlas_41_plus_ids.txt")
    ]
    
    DIR_OUT_WARPED = os.path.join(DATA_DIR, "warped_patients")
    
    register_cohort_to_template(
        template_path=FILE_TEMPLATE,
        negatives_dir=DIR_NEGATIVES,
        list_paths=LISTS,
        output_dir=DIR_OUT_WARPED
    )