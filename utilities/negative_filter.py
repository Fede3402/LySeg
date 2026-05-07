import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np

# --- CONFIGURAZIONE ---
# Percorso dove hai le maschere (lesion_mask_none.nii.gz)
CARTELLA_RISULTATI = Path("../Risultati_LION_Negativi") 
# Percorso dove hai le PET originali (Negative_Light)
CARTELLA_PET_ORIGINALI = Path("../Negative_Light")

SOGLIA_ML_MINIMA = 1.0 
SOGLIA_SUV_MINIMA = 4.0 

print(f" Avvio Filtro Combinato: > {SOGLIA_ML_MINIMA} mL E SUVmax > {SOGLIA_SUV_MINIMA}")

# Troviamo tutte le maschere
maschere = list(CARTELLA_RISULTATI.rglob('lesion_mask_none.nii.gz'))
fp_dopo = 0
totale = len(maschere)
print(totale)

for m in maschere:
    folder_name = m.parent.name
    pet_search = list(CARTELLA_PET_ORIGINALI.rglob(f"**/{folder_name}/SUV.nii*"))
    
    if not pet_search:
        print(f" PET non trovata per {folder_name}")
        continue
    
    img_pet = sitk.ReadImage(str(pet_search[0]))
    img_mask = sitk.ReadImage(str(m))

    # --- RESAMPLING (Stessa logica di prima per evitare l'IndexError) ---
    if img_mask.GetSize() != img_pet.GetSize():
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(img_pet)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetTransform(sitk.Transform())
        img_mask = resample.Execute(img_mask)

    # Parametri volumetrici
    spacing = img_mask.GetSpacing()
    vol_voxel_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
    soglia_voxel = int(round(SOGLIA_ML_MINIMA / vol_voxel_ml))

    # Componenti connesse
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled_img = cc_filter.Execute(img_mask)
    
    pet_array = sitk.GetArrayFromImage(img_pet)
    labeled_array = sitk.GetArrayFromImage(labeled_img)
    final_mask_array = np.zeros_like(labeled_array, dtype=np.uint8)
    
    num_isole = np.max(labeled_array)
    isole_mantenute = 0

    if num_isole > 0:
        for i in range(1, num_isole + 1):
            isola_coords = (labeled_array == i)
            volume_isola_voxel = np.sum(isola_coords)
            
            # Controllo SUVmax sull'isola
            suv_max_isola = np.max(pet_array[isola_coords])
            
            if volume_isola_voxel >= soglia_voxel and suv_max_isola >= SOGLIA_SUV_MINIMA:
                final_mask_array[isola_coords] = 1
                isole_mantenute += 1
            
    # Salvataggio
    cleaned_mask = sitk.GetImageFromArray(final_mask_array)
    cleaned_mask.CopyInformation(img_pet)
    
    final_path = m.parent / 'lesion_mask_combined_filter.nii.gz'
    sitk.WriteImage(cleaned_mask, str(final_path))
    
    if np.sum(final_mask_array) > 0:
        fp_dopo += 1
        print(f" {folder_name}: FP RIMASTO ({isole_mantenute} isole valide)")
    else:
        print(f"{folder_name}: PULITO")

print(f"\n🎯 Specificità FINALE: {((totale - fp_dopo)/totale)*100:.2f}%")