from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import SimpleITK as sitk
from medpy.metric.binary import hd95
from scipy.ndimage import label


# CONFIGURATION

DATASET_DIR   = Path("../Lymphoma")
RISULTATI_PRED = Path("../Risultati_LION")

SOGLIE_TESTATE = ["none", "mean", "median", "percist_1_5x", "stat_2sd", "fissa_4_0"]

results = []

# CREIAMO UN SET PER MEMORIZZARE GLI ID GIÀ VISTI 
pazienti_elaborati = set() 


for patient_folder in RISULTATI_PRED.iterdir():
    if not patient_folder.is_dir(): continue
    
    patient_id = patient_folder.name
    print(f'Analisi paziente: {patient_id}')

    
    # CONTROLLO ANTI-DUPLICATO
    if patient_id in pazienti_elaborati:
        continue
    # Aggiungiamo l'ID al set così non verrà ricalcolato
    pazienti_elaborati.add(patient_id)
    # -----------------------------------
    
    # Usiamo rglob per trovare "SEG.nii.gz" ovunque all'interno della cartella del paziente originale
    tutte_le_gt = list((DATASET_DIR / patient_id).rglob("SEG.nii.gz"))

    if not tutte_le_gt:
        print(f" Saltato paziente {patient_id}: Nessun SEG.nii.gz trovato nel dataset originale.")
        continue

    for gt_path in tutte_le_gt:
        # Estraiamo il nome della cartella genitore per capire di che scan si tratta
        scan_folder_name = gt_path.parent.name 
        
        # Creiamo un ID univoco unendo Paziente e Scan
        scan_id_univoco = f"{patient_id}_{scan_folder_name}"
        
        # CONTROLLO ANTI-DUPLICATO (Spostato qui e basato sull'ID univoco)
        if scan_id_univoco in pazienti_elaborati:
            continue
        pazienti_elaborati.add(scan_id_univoco)
        
        # --- CALCOLO VOLUME GT ---
        gt_img = sitk.ReadImage(str(gt_path))
        gt_array = sitk.GetArrayFromImage(gt_img) == 1
        spacing = gt_img.GetSpacing() 
        voxel_volume_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
        tmtv_gt_ml = np.sum(gt_array) * voxel_volume_ml

    
    # CERCHIAMO LE PREDIZIONI
        for soglia in SOGLIE_TESTATE:
            nome_file = f"lesion_mask_{soglia}.nii.gz"
            
            # Usiamo rglob così trova il file anche se è nascosto dentro sottocartelle
            pred_search = list((patient_folder / scan_folder_name).rglob(nome_file))
            
            if not pred_search:
                continue
                
            pred_path = pred_search[0]
            
            pred_img = sitk.ReadImage(str(pred_path))
            pred_array = sitk.GetArrayFromImage(pred_img) == 1
            
            tmtv_pred_ml = np.sum(pred_array) * voxel_volume_ml
            diff_relativa_tmtv = (abs(tmtv_gt_ml - tmtv_pred_ml)/tmtv_gt_ml)*100
            
            intersezione = np.sum(gt_array & pred_array)
            somma_gt = np.sum(gt_array)
            somma_pred = np.sum(pred_array)
            
            dice = 1.0 if (somma_gt + somma_pred) == 0 else (2.0 * intersezione) / (somma_gt + somma_pred)
            recall = intersezione / somma_gt if somma_gt > 0 else 1.0
            precision = intersezione / somma_pred if somma_pred > 0 else 1.0

            falsi_positivi_pixel = somma_pred - intersezione
            volume_falsi_positivi_ml = falsi_positivi_pixel * voxel_volume_ml
            fdr = 1.0 - precision # False Discovery Rate

            '''
            if somma_gt == 0 and somma_pred == 0:
                valore_hd95 = 0.0 # Entrambe vuote: match perfetto
            elif somma_gt == 0 or somma_pred == 0:
                valore_hd95 = np.nan # Una vuota e l'altra no: distanza non calcolabile. Usiamo NaN.
            else:
                # INVERSIONE SPACING: da (X,Y,Z) di SimpleITK a (Z,Y,X) di NumPy
                spacing_zyx = (spacing[2], spacing[1], spacing[0])
                
                # Calcolo della distanza in millimetri
                valore_hd95 = hd95(pred_array, gt_array, voxelspacing=spacing_zyx)
            '''
            fp_mask = pred_array & (~gt_array)
            structure = np.ones((3, 3, 3), dtype=int)
            labeled_array, num_fp_lesions = label(fp_mask, structure=structure)

            results.append({
                "Scan_ID": scan_id_univoco,
                "Soglia_Usata": soglia,
                "Dice_Score": dice,
                "Recall": recall,
                "Precision": precision,
                "FDR": fdr,
                "num_fp_lesions": num_fp_lesions,
                "Volume_FP_mL": volume_falsi_positivi_ml,
                "Errore_Volume_mL": diff_relativa_tmtv
            })



# --- SALVATAGGIO ---
df_risultati = pd.DataFrame(results)
percorso_csv = "../Metriche_Validazione.csv"
df_risultati.to_csv(percorso_csv, index=False)

print(f" Report salvato in: {percorso_csv}\n")

