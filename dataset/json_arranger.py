import os 
import pandas as pd 
import json 

########################### PATH CONFIGURATION ##########################
csv_path = "../data/Lymphoma_segmentation/Dynamic_Thresholds.csv"
json_output_path = "../data/Lymphoma_segmentation/dataset.json"

directory_splits = {
    
    "training":
    {
        "images":"../data/Lymphoma_segmentation/imagesTr",
        "labels":"../data/Lymphoma_segmentation/labelsTr"
    },

    "validation":
    {
        "images":"../data/Lymphoma_segmentation/imagesVal",
        "labels":"../data/Lymphoma_segmentation/labelsVal"
    },

    "testing":
    {
        "images":"../data/Lymphoma_segmentation/imagesTs",
        "labels":"../data/Lymphoma_segmentation/labelsTs"
    }

}

##########################################################################

def create_json():

    df = pd.read_csv(csv_path)

    suv_lookup = pd.Series(df["SUV_Mean"].values, index = df["Scan_ID"]).to_dict()

    dataset_dict = {
        "training" : [],
        "validation" : [],
        "testing" : []
    }

    patient_without_suv = 0

    # Iteration on every split 
    for split_name, path in directory_splits.items():
        img_dir = path["images"]
        lbl_dir = path["labels"]

        if not os.path.exists(img_dir):
            continue
            
        # 3. Leggiamo tutti i file nella cartella delle immagini
        for filename in os.listdir(img_dir):
            if filename.endswith("_0000.nii.gz"):
                
                # Estraiamo lo Scan_ID rimuovendo "_0000.nii.gz" (12 caratteri)
                scan_id = filename[:-12]
                
                # Costruiamo i percorsi completi
                pet_path = os.path.join(img_dir, filename)
                ct_path = os.path.join(img_dir, f"{scan_id}_0001.nii.gz")
                label_path = os.path.join(lbl_dir, f"{scan_id}.nii.gz")
                
                # 4. Cerchiamo il SUV nel nostro dizionario
                # Usiamo .get() così se un paziente manca nel CSV non crasha tutto, ma assegna None
                suv_mean = suv_lookup.get(scan_id, None)
                
                if suv_mean is None:
                    print(f"ATTENZIONE: Nessun valore SUV trovato nel CSV per {scan_id}")
                    pazienti_senza_suv += 1
                    continue # Decidiamo di saltare i pazienti senza SUV clinico
                
                # 5. Creiamo il dizionario per questo paziente e lo aggiungiamo allo split
                patient_data = {
                    "scan_id": scan_id,
                    "pet": pet_path,
                    "ct": ct_path,
                    "label": label_path,
                    "suv_prior": float(suv_mean)
                }
                
                dataset_dict[split_name].append(patient_data)

    # 6. Salviamo il file JSON finale
    with open(json_output_path, 'w') as f:
        json.dump(dataset_dict, f, indent=4) 

    print(" Generazione JSON completata!")
    print(f"    - Training:   {len(dataset_dict['training'])} pazienti")
    print(f"    - Validation: {len(dataset_dict['validation'])} pazienti")
    print(f"    - Test:       {len(dataset_dict['testing'])} pazienti")
    
    if patient_without_suv > 0:
        print(f"\n[!] Sono stati ignorati {patient_without_suv} pazienti presenti nelle cartelle ma mancanti nel CSV.")

if __name__ == "__main__":
    create_json()